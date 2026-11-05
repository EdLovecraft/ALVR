use alvr_common::{HEAD_ID, SlidingWindowAverage};
use alvr_events::{BitrateDirectives, GraphStatistics, StatisticsSummary};
use alvr_packets::{ClientStatistics, FrameTimingFeedback};
use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

const FULL_REPORT_INTERVAL: Duration = Duration::from_millis(500);
const EPS_INTERVAL: Duration = Duration::from_micros(1);

pub struct HistoryFrame {
    target_timestamp: Duration,
    tracking_received: Instant,
    frame_present: Instant,
    frame_composed: Instant,
    frame_encoded: Instant,
    video_packet_bytes: usize,
    total_pipeline_latency: Duration,
}

impl Default for HistoryFrame {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            target_timestamp: Duration::ZERO,
            tracking_received: now,
            frame_present: now,
            frame_composed: now,
            frame_encoded: now,
            video_packet_bytes: 0,
            total_pipeline_latency: Duration::ZERO,
        }
    }
}

#[derive(Default, Clone)]
struct BatteryData {
    gauge_value: f32,
    is_plugged: bool,
}

// Client frame timing feedback, summarized at the statistics summary rate.
#[derive(Default)]
struct ClientFrameTiming {
    latest: Option<FrameTimingFeedback>,
    // Feedback at the previous summary. Counters are totals, so rates are differences.
    summary_baseline: Option<FrameTimingFeedback>,
    phase_locked: bool,
}

struct FrameTimingSummary {
    repeated_fps: Option<f32>,
    dropped_fps: Option<f32>,
    latch_margin_low_ms: Option<f32>,
    latch_margin_median_ms: Option<f32>,
    phase_locked: bool,
}

impl ClientFrameTiming {
    fn summarize(&mut self) -> FrameTimingSummary {
        let latest = self.latest.as_ref();
        let rates = latest
            .zip(self.summary_baseline.as_ref())
            .filter(|(latest, baseline)| latest.client_time > baseline.client_time)
            .map(|(latest, baseline)| {
                let seconds = (latest.client_time - baseline.client_time).as_secs_f32();
                let rate =
                    |current: u64, previous: u64| current.saturating_sub(previous) as f32 / seconds;

                (
                    rate(latest.repeated_frames, baseline.repeated_frames),
                    rate(latest.dropped_frames, baseline.dropped_frames),
                )
            });

        let summary = FrameTimingSummary {
            repeated_fps: rates.map(|(repeated, _)| repeated),
            dropped_fps: rates.map(|(_, dropped)| dropped),
            latch_margin_low_ms: latest
                .and_then(|latest| latest.latch_margin_low_s)
                .map(|margin| margin * 1000.0),
            latch_margin_median_ms: latest
                .and_then(|latest| latest.latch_margin_median_s)
                .map(|margin| margin * 1000.0),
            phase_locked: self.phase_locked,
        };

        if let Some(latest) = latest {
            self.summary_baseline = Some(latest.clone());
        }

        summary
    }
}

// Owned snapshots let callers publish telemetry after releasing the statistics lock.
#[derive(Default)]
pub struct StatisticsReport {
    pub network_latency: Duration,
    pub game_latency: Duration,
    pub summary: Option<StatisticsSummary>,
    pub graph: Option<GraphStatistics>,
}

pub struct StatisticsManager {
    history_buffer: VecDeque<HistoryFrame>,
    max_history_size: usize,
    last_full_report_instant: Instant,
    last_frame_present_instant: Instant,
    last_frame_present_interval: Duration,
    video_packets_total: usize,
    video_packets_partial_sum: usize,
    video_bytes_total: usize,
    video_bytes_partial_sum: usize,
    battery_gauges: HashMap<u64, BatteryData>,
    steamvr_pipeline_latency: Duration,
    motion_to_photon_latency_average: SlidingWindowAverage<Duration>,
    last_throughput_directives: BitrateDirectives,
    client_frame_timing: ClientFrameTiming,
}

impl StatisticsManager {
    // history size used to calculate average total pipeline latency
    pub fn new(
        max_history_size: usize,
        nominal_server_frame_interval: Duration,
        steamvr_pipeline_frames: f32,
    ) -> Self {
        Self {
            history_buffer: VecDeque::new(),
            max_history_size,
            last_full_report_instant: Instant::now(),
            last_frame_present_instant: Instant::now(),
            last_frame_present_interval: Duration::ZERO,
            video_packets_total: 0,
            video_packets_partial_sum: 0,
            video_bytes_total: 0,
            video_bytes_partial_sum: 0,
            battery_gauges: HashMap::new(),
            steamvr_pipeline_latency: Duration::from_secs_f32(
                steamvr_pipeline_frames * nominal_server_frame_interval.as_secs_f32(),
            ),
            motion_to_photon_latency_average: SlidingWindowAverage::new(
                Duration::ZERO,
                max_history_size,
            ),
            last_throughput_directives: BitrateDirectives::default(),
            client_frame_timing: ClientFrameTiming::default(),
        }
    }

    pub fn report_tracking_received(&mut self, target_timestamp: Duration) {
        if !self
            .history_buffer
            .iter()
            .any(|frame| frame.target_timestamp == target_timestamp)
        {
            self.history_buffer.push_front(HistoryFrame {
                target_timestamp,
                tracking_received: Instant::now(),
                ..Default::default()
            });
        }

        if self.history_buffer.len() > self.max_history_size {
            self.history_buffer.pop_back();
        }
    }

    pub fn report_frame_present(&mut self, target_timestamp: Duration, offset: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            let now = Instant::now() - offset;

            self.last_frame_present_interval =
                now.saturating_duration_since(self.last_frame_present_instant);
            self.last_frame_present_instant = now;

            frame.frame_present = now;
        }
    }

    pub fn report_frame_composed(&mut self, target_timestamp: Duration, offset: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            frame.frame_composed = Instant::now() - offset;
        }
    }

    // returns encoding interval
    pub fn report_frame_encoded(
        &mut self,
        target_timestamp: Duration,
        bytes_count: usize,
    ) -> Duration {
        self.video_packets_total += 1;
        self.video_packets_partial_sum += 1;
        self.video_bytes_total += bytes_count;
        self.video_bytes_partial_sum += bytes_count;

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            frame.frame_encoded = Instant::now();

            frame.video_packet_bytes = bytes_count;

            frame
                .frame_encoded
                .saturating_duration_since(frame.frame_composed)
        } else {
            Duration::ZERO
        }
    }

    pub fn report_battery(&mut self, device_id: u64, gauge_value: f32, is_plugged: bool) {
        *self.battery_gauges.entry(device_id).or_default() = BatteryData {
            gauge_value,
            is_plugged,
        };
    }

    pub fn report_throughput_stats(&mut self, stats: BitrateDirectives) {
        self.last_throughput_directives = stats;
    }

    pub fn report_frame_timing(&mut self, feedback: FrameTimingFeedback, phase_locked: bool) {
        let timing = &mut self.client_frame_timing;

        // A new stream restarts the client totals.
        if timing
            .latest
            .as_ref()
            .is_some_and(|latest| feedback.client_time < latest.client_time)
        {
            timing.summary_baseline = None;
        }

        timing.latest = Some(feedback);
        timing.phase_locked = phase_locked;
    }

    // Called every frame. Publishing the returned snapshots must happen outside the lock.
    pub fn report_statistics(&mut self, client_stats: ClientStatistics) -> StatisticsReport {
        self.report_statistics_at(client_stats, Instant::now())
    }

    fn report_statistics_at(
        &mut self,
        client_stats: ClientStatistics,
        now: Instant,
    ) -> StatisticsReport {
        self.motion_to_photon_latency_average
            .submit_sample(client_stats.total_pipeline_latency);

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == client_stats.target_timestamp)
        {
            frame.total_pipeline_latency = client_stats.total_pipeline_latency;

            let game_time_latency = frame
                .frame_present
                .saturating_duration_since(frame.tracking_received);

            let server_compositor_latency = frame
                .frame_composed
                .saturating_duration_since(frame.frame_present);

            let encoder_latency = frame
                .frame_encoded
                .saturating_duration_since(frame.frame_composed);

            // The network latency cannot be estiamed directly. It is what's left of the total
            // latency after subtracting all other latency intervals. In particular it contains the
            // transport latency of the tracking packet and the interval between the first video
            // packet is sent and the last video packet is received for a specific frame.
            // For safety, use saturating_sub to avoid a crash if for some reason the network
            // latency is miscalculated as negative.
            let network_latency = frame.total_pipeline_latency.saturating_sub(
                game_time_latency
                    + server_compositor_latency
                    + encoder_latency
                    + client_stats.video_decode
                    + client_stats.video_decoder_queue
                    + client_stats.rendering
                    + client_stats.vsync_queue,
            );

            let client_fps =
                1.0 / Duration::max(client_stats.frame_interval, EPS_INTERVAL).as_secs_f32();
            let server_fps =
                1.0 / Duration::max(self.last_frame_present_interval, EPS_INTERVAL).as_secs_f32();

            let summary = if self.last_full_report_instant + FULL_REPORT_INTERVAL < now {
                self.last_full_report_instant += FULL_REPORT_INTERVAL;

                let interval_secs = FULL_REPORT_INTERVAL.as_secs_f32();
                let frame_timing = self.client_frame_timing.summarize();

                let summary = StatisticsSummary {
                    video_packets_total: self.video_packets_total,
                    video_packets_per_sec: (self.video_packets_partial_sum as f32 / interval_secs)
                        as _,
                    video_mbytes_total: (self.video_bytes_total as f32 / 1e6) as usize,
                    video_mbits_per_sec: self.video_bytes_partial_sum as f32 * 8.
                        / 1e6
                        / interval_secs,
                    total_latency_ms: client_stats.total_pipeline_latency.as_secs_f32() * 1000.,
                    network_latency_ms: network_latency.as_secs_f32() * 1000.,
                    encode_latency_ms: encoder_latency.as_secs_f32() * 1000.,
                    decode_latency_ms: client_stats.video_decode.as_secs_f32() * 1000.,
                    client_fps: client_fps as _,
                    server_fps: server_fps as _,
                    battery_hmd: (self
                        .battery_gauges
                        .get(&HEAD_ID)
                        .cloned()
                        .unwrap_or_default()
                        .gauge_value
                        * 100.) as u32,
                    hmd_plugged: self
                        .battery_gauges
                        .get(&HEAD_ID)
                        .cloned()
                        .unwrap_or_default()
                        .is_plugged,
                    client_repeated_fps: frame_timing.repeated_fps,
                    client_dropped_fps: frame_timing.dropped_fps,
                    latch_margin_low_ms: frame_timing.latch_margin_low_ms,
                    latch_margin_median_ms: frame_timing.latch_margin_median_ms,
                    frame_pacing_phase_locked: frame_timing.phase_locked,
                };

                self.video_packets_partial_sum = 0;
                self.video_bytes_partial_sum = 0;

                Some(summary)
            } else {
                None
            };

            let packet_bits = frame.video_packet_bytes as f32 * 8.0;
            let throughput_bps =
                packet_bits / Duration::max(network_latency, EPS_INTERVAL).as_secs_f32();
            let bitrate_bps = packet_bits
                / Duration::max(self.last_frame_present_interval, EPS_INTERVAL).as_secs_f32();

            // todo: use target timestamp in nanoseconds. the dashboard needs to use the first
            // timestamp as the graph time origin.
            let graph = GraphStatistics {
                total_pipeline_latency_s: client_stats.total_pipeline_latency.as_secs_f32(),
                game_time_s: game_time_latency.as_secs_f32(),
                server_compositor_s: server_compositor_latency.as_secs_f32(),
                encoder_s: encoder_latency.as_secs_f32(),
                network_s: network_latency.as_secs_f32(),
                decoder_s: client_stats.video_decode.as_secs_f32(),
                decoder_queue_s: client_stats.video_decoder_queue.as_secs_f32(),
                client_compositor_s: client_stats.rendering.as_secs_f32(),
                vsync_queue_s: client_stats.vsync_queue.as_secs_f32(),
                client_fps,
                server_fps,
                bitrate_directives: self.last_throughput_directives.clone(),
                throughput_bps,
                bitrate_bps,
            };

            StatisticsReport {
                network_latency,
                game_latency: game_time_latency,
                summary,
                graph: Some(graph),
            }
        } else {
            StatisticsReport::default()
        }
    }

    pub fn motion_to_photon_latency_average(&self) -> Duration {
        self.motion_to_photon_latency_average.get_average()
    }

    pub fn tracker_pose_time_offset(&self) -> Duration {
        // This is the opposite of the client's StatisticsManager::tracker_prediction_offset().
        self.steamvr_pipeline_latency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame() -> (StatisticsManager, ClientStatistics, Instant) {
        let start = Instant::now();
        let timestamp = Duration::from_secs(1);
        let mut stats = StatisticsManager::new(16, Duration::from_millis(10), 1.0);
        stats.last_full_report_instant = start;
        stats.last_frame_present_interval = Duration::from_millis(20);
        stats.history_buffer.push_front(HistoryFrame {
            target_timestamp: timestamp,
            tracking_received: start,
            frame_present: start + Duration::from_millis(4),
            frame_composed: start + Duration::from_millis(6),
            frame_encoded: start + Duration::from_millis(9),
            video_packet_bytes: 62_500,
            total_pipeline_latency: Duration::ZERO,
        });
        stats.report_throughput_stats(BitrateDirectives {
            requested_bitrate_bps: 30_000_000.0,
            ..Default::default()
        });
        stats.report_battery(*HEAD_ID, 0.75, true);

        let client = ClientStatistics {
            target_timestamp: timestamp,
            frame_interval: Duration::from_millis(10),
            video_decode: Duration::from_millis(5),
            video_decoder_queue: Duration::from_millis(2),
            rendering: Duration::from_millis(1),
            vsync_queue: Duration::from_millis(3),
            total_pipeline_latency: Duration::from_millis(30),
        };

        (stats, client, start)
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn matching_frame_returns_owned_latency_and_graph_snapshots() {
        let (mut stats, client, start) = sample_frame();
        let report = stats.report_statistics_at(client, start);

        assert_eq!(report.network_latency, Duration::from_millis(10));
        assert_eq!(report.game_latency, Duration::from_millis(4));
        assert!(report.summary.is_none());
        assert_eq!(
            stats.history_buffer[0].total_pipeline_latency,
            Duration::from_millis(30)
        );

        // Later updates must not change an event awaiting publication outside the lock.
        stats.report_throughput_stats(BitrateDirectives::default());
        let graph = report.graph.unwrap();
        assert_close(graph.total_pipeline_latency_s, 0.030);
        assert_close(graph.game_time_s, 0.004);
        assert_close(graph.server_compositor_s, 0.002);
        assert_close(graph.encoder_s, 0.003);
        assert_close(graph.network_s, 0.010);
        assert_close(graph.decoder_s, 0.005);
        assert_close(graph.decoder_queue_s, 0.002);
        assert_close(graph.client_compositor_s, 0.001);
        assert_close(graph.vsync_queue_s, 0.003);
        assert_close(graph.client_fps, 100.0);
        assert_close(graph.server_fps, 50.0);
        assert_close(graph.throughput_bps, 50_000_000.0);
        assert_close(graph.bitrate_bps, 25_000_000.0);
        assert_close(graph.bitrate_directives.requested_bitrate_bps, 30_000_000.0);
    }

    #[test]
    fn missing_frame_updates_prediction_without_publishing_or_resetting_counters() {
        let (mut stats, mut client, start) = sample_frame();
        client.target_timestamp += Duration::from_secs(1);
        stats.video_packets_partial_sum = 2;
        stats.video_bytes_partial_sum = 125_000;

        let report = stats.report_statistics_at(client, start + FULL_REPORT_INTERVAL * 2);

        assert_eq!(report.network_latency, Duration::ZERO);
        assert_eq!(report.game_latency, Duration::ZERO);
        assert!(report.summary.is_none());
        assert!(report.graph.is_none());
        // The initial zero sample and the new 30 ms sample are both retained.
        assert_eq!(
            stats.motion_to_photon_latency_average(),
            Duration::from_millis(15)
        );
        assert_eq!(stats.video_packets_partial_sum, 2);
        assert_eq!(stats.video_bytes_partial_sum, 125_000);
        assert_eq!(stats.last_full_report_instant, start);
    }

    #[test]
    fn summary_snapshots_keep_the_reporting_period_and_reset_only_partial_counts() {
        let (mut stats, client, start) = sample_frame();
        stats.video_packets_total = 20;
        stats.video_packets_partial_sum = 2;
        stats.video_bytes_total = 1_250_000;
        stats.video_bytes_partial_sum = 125_000;

        let boundary = start + FULL_REPORT_INTERVAL;
        assert!(
            stats
                .report_statistics_at(client.clone(), boundary)
                .summary
                .is_none()
        );
        assert_eq!(stats.video_packets_partial_sum, 2);

        let report = stats.report_statistics_at(client.clone(), boundary + EPS_INTERVAL);
        let summary = report.summary.unwrap();
        assert_eq!(summary.video_packets_total, 20);
        assert_eq!(summary.video_packets_per_sec, 4);
        assert_eq!(summary.video_mbytes_total, 1);
        assert_close(summary.video_mbits_per_sec, 2.0);
        assert_close(summary.total_latency_ms, 30.0);
        assert_close(summary.network_latency_ms, 10.0);
        assert_close(summary.encode_latency_ms, 3.0);
        assert_close(summary.decode_latency_ms, 5.0);
        assert_eq!(summary.client_fps, 100);
        assert_eq!(summary.server_fps, 50);
        assert_eq!(summary.battery_hmd, 75);
        assert!(summary.hmd_plugged);
        assert_eq!(stats.video_packets_total, 20);
        assert_eq!(stats.video_bytes_total, 1_250_000);
        assert_eq!(stats.video_packets_partial_sum, 0);
        assert_eq!(stats.video_bytes_partial_sum, 0);
        assert_eq!(stats.last_full_report_instant, boundary);

        let next_boundary = boundary + FULL_REPORT_INTERVAL;
        assert!(
            stats
                .report_statistics_at(client.clone(), next_boundary)
                .summary
                .is_none()
        );
        let next_summary = stats
            .report_statistics_at(client, next_boundary + EPS_INTERVAL)
            .summary
            .unwrap();
        assert_eq!(next_summary.video_packets_total, 20);
        assert_eq!(next_summary.video_packets_per_sec, 0);
        assert_close(next_summary.video_mbits_per_sec, 0.0);
    }

    fn frame_timing(client_ms: u64, repeated: u64, dropped: u64) -> FrameTimingFeedback {
        FrameTimingFeedback {
            client_time: Duration::from_millis(client_ms),
            display_period: Duration::from_millis(10),
            latched_frames: 0,
            repeated_frames: repeated,
            dropped_frames: dropped,
            latch_margin_low_s: Some(0.0015),
            latch_margin_median_s: Some(0.004),
        }
    }

    #[test]
    fn summaries_report_client_frame_timing() {
        let (mut stats, client, start) = sample_frame();

        // Without a baseline only the margins and the lock state are known.
        stats.report_frame_timing(frame_timing(1000, 10, 2), true);
        let first = stats
            .report_statistics_at(client.clone(), start + FULL_REPORT_INTERVAL + EPS_INTERVAL)
            .summary
            .unwrap();
        assert_eq!(first.client_repeated_fps, None);
        assert_eq!(first.client_dropped_fps, None);
        assert_close(first.latch_margin_low_ms.unwrap(), 1.5);
        assert_close(first.latch_margin_median_ms.unwrap(), 4.0);
        assert!(first.frame_pacing_phase_locked);

        // Five repeats and one drop over 500 ms of client time.
        stats.report_frame_timing(frame_timing(1500, 15, 3), false);
        let second = stats
            .report_statistics_at(client, start + FULL_REPORT_INTERVAL * 2 + EPS_INTERVAL)
            .summary
            .unwrap();
        assert_close(second.client_repeated_fps.unwrap(), 10.0);
        assert_close(second.client_dropped_fps.unwrap(), 2.0);
        assert!(!second.frame_pacing_phase_locked);
    }

    #[test]
    fn summaries_without_client_frame_timing_report_none() {
        let (mut stats, client, start) = sample_frame();

        let summary = stats
            .report_statistics_at(client, start + FULL_REPORT_INTERVAL + EPS_INTERVAL)
            .summary
            .unwrap();

        assert_eq!(summary.client_repeated_fps, None);
        assert_eq!(summary.latch_margin_low_ms, None);
        assert!(!summary.frame_pacing_phase_locked);
    }
}
