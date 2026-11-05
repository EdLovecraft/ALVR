use alvr_common::SlidingWindowAverage;
use alvr_packets::{ClientStatistics, FrameTimingFeedback};
use std::{
    collections::VecDeque,
    mem,
    time::{Duration, Instant},
};

const FRAME_TIMING_INTERVAL: Duration = Duration::from_millis(250);
const LATCH_MARGIN_WINDOW: Duration = Duration::from_secs(1);
// A low percentile rather than the minimum: a single outlier must not set the pacing phase.
const LATCH_MARGIN_LOW_QUANTILE: f32 = 0.02;
const MIN_LATCH_MARGIN_SAMPLES: usize = 20;
// A frame shown after a longer run of repeated refreshes (stream start, IDR recovery, network
// outage) says nothing about the pacing phase.
const MAX_REPEATS_BEFORE_LATCH_SAMPLE: u32 = 3;

/// Outcome of one display refresh: whether a new video frame was latched before the deadline.
pub struct FrameLatch {
    /// End of the wait for a decoded frame in this refresh.
    pub deadline: Instant,
    /// Frame latched for display, if any.
    pub frame_timestamp: Option<Duration>,
    pub display_period: Duration,
    /// Decoded frames discarded since the previous refresh.
    pub dropped_frames: u32,
}

struct HistoryFrame {
    input_acquired: Instant,
    video_packet_received: Instant,
    frame_decoded: Option<Instant>,
    client_stats: ClientStatistics,
}

struct LatchTracker {
    stream_start: Instant,
    next_feedback: Instant,
    // Deadline of the first refresh after the last latched frame. A frame that missed it is
    // measured against it, so that "slightly late" doesn't look like "very early".
    first_missed_deadline: Option<Instant>,
    consecutive_repeats: u32,
    margins: VecDeque<(Instant, f32)>,
    display_period: Duration,
    latched_frames: u64,
    repeated_frames: u64,
    dropped_frames: u64,
}

impl LatchTracker {
    fn new(now: Instant) -> Self {
        Self {
            stream_start: now,
            next_feedback: now + FRAME_TIMING_INTERVAL,
            first_missed_deadline: None,
            consecutive_repeats: 0,
            margins: VecDeque::new(),
            display_period: Duration::ZERO,
            latched_frames: 0,
            repeated_frames: 0,
            dropped_frames: 0,
        }
    }
}

// Positive if `deadline` comes after `ready`.
fn signed_seconds(deadline: Instant, ready: Instant) -> f32 {
    if deadline >= ready {
        (deadline - ready).as_secs_f32()
    } else {
        -(ready - deadline).as_secs_f32()
    }
}

fn quantile(sorted: &[f32], quantile: f32) -> f32 {
    sorted[((sorted.len() as f32 * quantile) as usize).min(sorted.len() - 1)]
}

pub struct StatisticsManager {
    history_buffer: VecDeque<HistoryFrame>,
    max_history_size: usize,
    prev_vsync: Instant,
    total_pipeline_latency_average: SlidingWindowAverage<Duration>,
    latch: LatchTracker,
}

impl StatisticsManager {
    pub fn new(max_history_size: usize) -> Self {
        Self {
            max_history_size,
            history_buffer: VecDeque::new(),
            prev_vsync: Instant::now(),
            total_pipeline_latency_average: SlidingWindowAverage::new(
                Duration::ZERO,
                max_history_size,
            ),
            latch: LatchTracker::new(Instant::now()),
        }
    }

    pub fn report_input_acquired(&mut self, target_timestamp: Duration) {
        if !self
            .history_buffer
            .iter()
            .any(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            self.history_buffer.push_front(HistoryFrame {
                input_acquired: Instant::now(),
                // this is just a placeholder because Instant does not have a default value
                video_packet_received: Instant::now(),
                frame_decoded: None,
                client_stats: ClientStatistics {
                    target_timestamp,
                    ..Default::default()
                },
            });
        }

        if self.history_buffer.len() > self.max_history_size {
            self.history_buffer.pop_back();
        }
    }

    pub fn report_video_packet_received(&mut self, target_timestamp: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            frame.video_packet_received = Instant::now();
        }
    }

    pub fn report_frame_decoded(&mut self, target_timestamp: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            let now = Instant::now();
            frame.frame_decoded = Some(now);
            frame.client_stats.video_decode =
                now.saturating_duration_since(frame.video_packet_received);
        }
    }

    pub fn report_compositor_start(&mut self, target_timestamp: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            frame.client_stats.video_decoder_queue = Instant::now().saturating_duration_since(
                frame.video_packet_received + frame.client_stats.video_decode,
            );
        }
    }

    // Called once per display refresh while streaming. Returns a feedback snapshot for the
    // streamer every FRAME_TIMING_INTERVAL.
    pub fn report_frame_latch(&mut self, latch: FrameLatch) -> Option<FrameTimingFeedback> {
        self.report_frame_latch_at(latch, Instant::now())
    }

    fn report_frame_latch_at(
        &mut self,
        latch: FrameLatch,
        now: Instant,
    ) -> Option<FrameTimingFeedback> {
        let tracker = &mut self.latch;
        tracker.dropped_frames += u64::from(latch.dropped_frames);
        tracker.display_period = latch.display_period;

        if let Some(timestamp) = latch.frame_timestamp {
            tracker.latched_frames += 1;

            let reference_deadline = tracker
                .first_missed_deadline
                .take()
                .unwrap_or(latch.deadline);
            let repeats = mem::take(&mut tracker.consecutive_repeats);
            let ready = self
                .history_buffer
                .iter()
                .find(|frame| frame.client_stats.target_timestamp == timestamp)
                .and_then(|frame| frame.frame_decoded);

            if repeats <= MAX_REPEATS_BEFORE_LATCH_SAMPLE
                && !latch.display_period.is_zero()
                && let Some(ready) = ready
            {
                // Bound outliers, for example a frame that waited behind a long backlog.
                let bound = 2.0 * latch.display_period.as_secs_f32();
                let margin = signed_seconds(reference_deadline, ready);
                tracker
                    .margins
                    .push_back((now, margin.clamp(-bound, bound)));
            }
        } else {
            tracker.repeated_frames += 1;
            tracker.consecutive_repeats += 1;

            // Before the first frame there is no stream to be late for.
            if tracker.latched_frames > 0 {
                tracker.first_missed_deadline.get_or_insert(latch.deadline);
            }
        }

        while tracker
            .margins
            .front()
            .is_some_and(|(time, _)| now.saturating_duration_since(*time) > LATCH_MARGIN_WINDOW)
        {
            tracker.margins.pop_front();
        }

        if now < tracker.next_feedback {
            return None;
        }
        tracker.next_feedback = now + FRAME_TIMING_INTERVAL;

        let (latch_margin_low_s, latch_margin_median_s) =
            if tracker.margins.len() >= MIN_LATCH_MARGIN_SAMPLES {
                let mut sorted = tracker
                    .margins
                    .iter()
                    .map(|(_, margin)| *margin)
                    .collect::<Vec<_>>();
                sorted.sort_by(f32::total_cmp);

                (
                    Some(quantile(&sorted, LATCH_MARGIN_LOW_QUANTILE)),
                    Some(quantile(&sorted, 0.5)),
                )
            } else {
                (None, None)
            };

        Some(FrameTimingFeedback {
            client_time: now.saturating_duration_since(tracker.stream_start),
            display_period: tracker.display_period,
            latched_frames: tracker.latched_frames,
            repeated_frames: tracker.repeated_frames,
            dropped_frames: tracker.dropped_frames,
            latch_margin_low_s,
            latch_margin_median_s,
        })
    }

    // vsync_queue is the latency between this call and the vsync. it cannot be measured by ALVR and
    // should be reported by the VR runtime
    pub fn report_submit(&mut self, target_timestamp: Duration, vsync_queue: Duration) {
        let now = Instant::now();

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            frame.client_stats.rendering = now.saturating_duration_since(
                frame.video_packet_received
                    + frame.client_stats.video_decode
                    + frame.client_stats.video_decoder_queue,
            );
            frame.client_stats.vsync_queue = vsync_queue;
            frame.client_stats.total_pipeline_latency =
                now.saturating_duration_since(frame.input_acquired) + vsync_queue;
            self.total_pipeline_latency_average
                .submit_sample(frame.client_stats.total_pipeline_latency);

            let vsync = now + vsync_queue;
            frame.client_stats.frame_interval = vsync.saturating_duration_since(self.prev_vsync);
            self.prev_vsync = vsync;
        }
    }

    pub fn summary(&self, target_timestamp: Duration) -> Option<ClientStatistics> {
        self.history_buffer
            .iter()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
            .map(|frame| frame.client_stats.clone())
    }

    // latency used for head prediction
    pub fn average_total_pipeline_latency(&self) -> Duration {
        self.total_pipeline_latency_average.get_average()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: Duration = Duration::from_millis(10);

    struct Harness {
        stats: StatisticsManager,
        start: Instant,
        next_timestamp: u64,
    }

    impl Harness {
        fn new() -> Self {
            let start = Instant::now();
            let mut stats = StatisticsManager::new(256);
            stats.latch = LatchTracker::new(start);

            Self {
                stats,
                start,
                next_timestamp: 1,
            }
        }

        // Registers a decoded frame that became ready at `ready_ms` after the start.
        fn decoded_frame(&mut self, ready_ms: u64) -> Duration {
            let timestamp = Duration::from_millis(self.next_timestamp);
            self.next_timestamp += 1;

            self.stats.report_input_acquired(timestamp);
            let frame = self
                .stats
                .history_buffer
                .iter_mut()
                .find(|frame| frame.client_stats.target_timestamp == timestamp)
                .unwrap();
            frame.frame_decoded = Some(self.start + Duration::from_millis(ready_ms));

            timestamp
        }

        // One refresh with its latch deadline at `deadline_ms`.
        fn latch(
            &mut self,
            deadline_ms: u64,
            frame_timestamp: Option<Duration>,
            dropped_frames: u32,
        ) -> Option<FrameTimingFeedback> {
            let deadline = self.start + Duration::from_millis(deadline_ms);
            self.stats.report_frame_latch_at(
                FrameLatch {
                    deadline,
                    frame_timestamp,
                    display_period: PERIOD,
                    dropped_frames,
                },
                deadline,
            )
        }

        fn margins(&self) -> Vec<f32> {
            self.stats
                .latch
                .margins
                .iter()
                .map(|(_, margin)| *margin)
                .collect()
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn on_time_frames_report_their_margin_before_the_deadline() {
        let mut harness = Harness::new();

        let frame = harness.decoded_frame(6);
        harness.latch(10, Some(frame), 0);

        assert_eq!(harness.margins().len(), 1);
        assert_close(harness.margins()[0], 0.004);
    }

    #[test]
    fn a_frame_after_a_repeat_is_measured_against_the_missed_deadline() {
        let mut harness = Harness::new();

        let first = harness.decoded_frame(5);
        harness.latch(10, Some(first), 0);
        // Ready 1 ms after the deadline at 20 ms, so it can only be shown at the next refresh.
        let second = harness.decoded_frame(21);
        harness.latch(20, None, 0);
        harness.latch(30, Some(second), 0);

        assert_close(harness.margins()[1], -0.001);
        assert_eq!(harness.stats.latch.repeated_frames, 1);
        assert_eq!(harness.stats.latch.latched_frames, 2);
    }

    #[test]
    fn refreshes_before_the_first_frame_are_not_missed_deadlines() {
        let mut harness = Harness::new();

        // The stream has not started yet: nothing was late for these refreshes.
        harness.latch(10, None, 0);
        harness.latch(20, None, 0);
        let frame = harness.decoded_frame(26);
        harness.latch(30, Some(frame), 0);

        assert_close(harness.margins()[0], 0.004);
        assert_eq!(harness.stats.latch.repeated_frames, 2);
    }

    #[test]
    fn a_backlogged_frame_reports_the_extra_refresh_it_waited() {
        let mut harness = Harness::new();

        let first = harness.decoded_frame(5);
        let second = harness.decoded_frame(6);
        harness.latch(10, Some(first), 0);
        harness.latch(20, Some(second), 0);

        // The second frame could have become ready up to 14 ms later and still be shown.
        assert_close(harness.margins()[1], 0.014);
    }

    #[test]
    fn frames_after_a_long_outage_are_not_sampled() {
        let mut harness = Harness::new();

        let frame = harness.decoded_frame(5);
        harness.latch(10, Some(frame), 0);
        for refresh in 2..=(MAX_REPEATS_BEFORE_LATCH_SAMPLE + 2) {
            harness.latch(u64::from(refresh) * 10, None, 0);
        }
        let frame = harness.decoded_frame(59);
        harness.latch(60, Some(frame), 0);

        assert_eq!(harness.margins().len(), 1);
        // The outage is still counted, and the next frame is sampled normally.
        assert_eq!(harness.stats.latch.repeated_frames, 4);
        let frame = harness.decoded_frame(65);
        harness.latch(70, Some(frame), 0);
        assert_close(harness.margins()[1], 0.005);
    }

    #[test]
    fn feedback_reports_totals_and_percentiles_periodically() {
        let mut harness = Harness::new();
        let mut reports = vec![];

        // Refreshes every 10 ms. Margins of 1..=30 ms, one dropped frame each.
        for refresh in 1..=30 {
            let frame = harness.decoded_frame(refresh * 10 - refresh);
            reports.extend(harness.latch(refresh * 10, Some(frame), 1));
        }
        // A repeat, then a frame that missed it by 2 ms, then margins of 3 ms.
        reports.extend(harness.latch(310, None, 0));
        let frame = harness.decoded_frame(312);
        reports.extend(harness.latch(320, Some(frame), 0));
        for deadline_ms in (330..=500).step_by(10) {
            let frame = harness.decoded_frame(deadline_ms - 3);
            reports.extend(harness.latch(deadline_ms, Some(frame), 0));
        }

        // One report every 250 ms.
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].client_time, Duration::from_millis(250));
        assert_eq!(reports[0].latched_frames, 25);
        assert_eq!(reports[0].dropped_frames, 25);
        assert_eq!(reports[0].repeated_frames, 0);

        let feedback = &reports[1];
        assert_eq!(feedback.client_time, Duration::from_millis(500));
        assert_eq!(feedback.display_period, PERIOD);
        assert_eq!(feedback.latched_frames, 49);
        assert_eq!(feedback.repeated_frames, 1);
        assert_eq!(feedback.dropped_frames, 30);
        // 49 samples: -2, 1, 2, nineteen of 3, then 4..=30 ms.
        assert_close(feedback.latch_margin_low_s.unwrap(), -0.002);
        assert_close(feedback.latch_margin_median_s.unwrap(), 0.006);
    }

    #[test]
    fn latch_margins_cover_only_the_last_second() {
        let mut harness = Harness::new();

        for deadline_ms in (10..=1500).step_by(10) {
            let frame = harness.decoded_frame(deadline_ms - 5);
            harness.latch(deadline_ms, Some(frame), 0);
        }

        // Samples from 500 ms to 1500 ms.
        assert_eq!(harness.margins().len(), 101);
    }

    #[test]
    fn too_few_samples_report_no_percentiles() {
        let mut harness = Harness::new();

        let frame = harness.decoded_frame(5);
        harness.latch(10, Some(frame), 0);
        let feedback = harness.latch(260, None, 0).unwrap();

        assert_eq!(feedback.latch_margin_low_s, None);
        assert_eq!(feedback.latch_margin_median_s, None);
        assert_eq!(feedback.latched_frames, 1);
        assert_eq!(feedback.repeated_frames, 1);
    }
}
