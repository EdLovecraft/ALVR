use std::time::{Duration, Instant};

// Client feedback older than this no longer describes the display phase: pace without it.
const PHASE_LOCK_TIMEOUT: Duration = Duration::from_secs(2);
// Fraction of the measured margin error corrected per feedback report (about four per second).
// Low values filter latch noise. Refining the frame interval with the display period removes most
// of the clock drift a low gain would otherwise have to follow.
const PHASE_LOCK_GAIN: f64 = 0.1;
// Largest grid movement per frame, so the cadence SteamVR observes changes by about 1% at most.
const MAX_PHASE_STEP: Duration = Duration::from_micros(100);
// While most frames miss their slot, the game and not the grid sets the cadence. Corrections
// would only rotate the grid without effect.
const MAX_LATE_FRACTION_FOR_CORRECTION: f32 = 0.5;
// The display period refines the negotiated frame interval. It never replaces a different rate.
const MAX_DISPLAY_PERIOD_DEVIATION: f64 = 0.01;

// This state has its own lock so statistics processing cannot delay the pacing deadline.
//
// Without client feedback the grid is free running: its phase relative to the headset display is
// arbitrary, and every late frame moves it later. With feedback (phase lock), the grid keeps its
// phase through late frames and is steered so that frames become ready shortly before the client
// latches them.
pub struct FramePacer {
    last_vsync_time: Instant,
    nominal_frame_interval: Duration,
    frame_interval: Duration,
    // Signed grid correction still to apply, in nanoseconds. Positive values delay the grid.
    pending_phase_shift_ns: i64,
    last_feedback: Option<Instant>,
    last_claim: Option<Instant>,
    claims_since_feedback: u32,
    late_claims_since_feedback: u32,
}

impl FramePacer {
    pub fn new(frame_interval: Duration) -> Self {
        Self::new_at(frame_interval, Instant::now())
    }

    fn new_at(frame_interval: Duration, now: Instant) -> Self {
        Self {
            last_vsync_time: now,
            nominal_frame_interval: frame_interval,
            frame_interval,
            pending_phase_shift_ns: 0,
            last_feedback: None,
            last_claim: None,
            claims_since_feedback: 0,
            late_claims_since_feedback: 0,
        }
    }

    // NB: this call is non-blocking, waiting should be done externally
    pub fn claim_next_vsync_slot(&mut self, throttle_frames: u32) -> Duration {
        self.claim_next_vsync_slot_at(throttle_frames, Instant::now())
    }

    fn claim_next_vsync_slot_at(&mut self, throttle_frames: u32, now: Instant) -> Duration {
        let phase_locked = self.feedback_is_recent(now);
        if phase_locked {
            self.apply_phase_step();
        }

        let wait = claim_next_vsync_slot(
            &mut self.last_vsync_time,
            self.frame_interval,
            throttle_frames,
            now,
            phase_locked,
        );

        self.last_claim = Some(now);
        self.claims_since_feedback = self.claims_since_feedback.saturating_add(1);
        if wait.is_zero() {
            self.late_claims_since_feedback = self.late_claims_since_feedback.saturating_add(1);
        }

        wait
    }

    /// Applies a client frame timing report. `latch_margin_low` is the client's low percentile of
    /// the time between frames becoming ready and its latch deadline, in seconds.
    /// `target_margin` is None while phase lock is disabled.
    pub fn report_client_timing(
        &mut self,
        display_period: Duration,
        latch_margin_low: Option<f32>,
        target_margin: Option<Duration>,
        now: Instant,
    ) {
        let late_fraction = if self.claims_since_feedback > 0 {
            self.late_claims_since_feedback as f32 / self.claims_since_feedback as f32
        } else {
            0.0
        };
        self.claims_since_feedback = 0;
        self.late_claims_since_feedback = 0;

        let nominal_secs = self.nominal_frame_interval.as_secs_f64();
        let deviation = (display_period.as_secs_f64() - nominal_secs).abs();

        // At a different refresh rate than negotiated, frames and refreshes have no fixed phase
        // relation to lock to: corrections would only rotate the grid.
        let Some(target_margin) =
            target_margin.filter(|_| deviation <= nominal_secs * MAX_DISPLAY_PERIOD_DEVIATION)
        else {
            self.last_feedback = None;
            self.pending_phase_shift_ns = 0;
            self.frame_interval = self.nominal_frame_interval;

            return;
        };

        self.frame_interval = display_period;

        // Too few latched frames for a percentile. Without fresh feedback the lock times out.
        let Some(latch_margin_low) = latch_margin_low else {
            return;
        };
        self.last_feedback = Some(now);

        if late_fraction <= MAX_LATE_FRACTION_FOR_CORRECTION {
            let error_ns =
                (f64::from(latch_margin_low) - target_margin.as_secs_f64()) * 1e9 * PHASE_LOCK_GAIN;
            // Half a frame either way covers any phase; more would only wind up.
            let limit = (self.frame_interval.as_nanos() / 2) as i64;
            self.pending_phase_shift_ns =
                (self.pending_phase_shift_ns + error_ns.round() as i64).clamp(-limit, limit);
        }
    }

    /// Whether frames are currently being paced against client feedback.
    pub fn is_phase_locked(&self, now: Instant) -> bool {
        self.feedback_is_recent(now)
            && self
                .last_claim
                .is_some_and(|claim| now.saturating_duration_since(claim) < PHASE_LOCK_TIMEOUT)
    }

    fn feedback_is_recent(&self, now: Instant) -> bool {
        self.last_feedback
            .is_some_and(|feedback| now.saturating_duration_since(feedback) < PHASE_LOCK_TIMEOUT)
    }

    fn apply_phase_step(&mut self) {
        let max_step = MAX_PHASE_STEP.as_nanos() as i64;
        let step = self.pending_phase_shift_ns.clamp(-max_step, max_step);

        let shifted = if step >= 0 {
            self.last_vsync_time
                .checked_add(Duration::from_nanos(step as u64))
        } else {
            self.last_vsync_time
                .checked_sub(Duration::from_nanos(step.unsigned_abs()))
        };

        if let Some(shifted) = shifted {
            self.last_vsync_time = shifted;
            self.pending_phase_shift_ns -= step;
        }
    }
}

// Claims the next vsync slot and returns how long the caller must wait for it. This is not a pure
// query: every call advances the pacing phase by one frame interval (plus any extra frames SteamVR
// asked to throttle by), so it must be called exactly once per produced frame, and only while
// server frame pacing is enabled.
//
// A frame that arrives after its slot has passed is let through immediately, which keeps
// framerates below the refresh rate from being quantized to an integer divisor of it (a slightly
// late 90 Hz frame would otherwise be delayed to the 45 Hz slot). Without `keep_phase`, the late
// frame re-anchors the grid, so the next frame gets a whole interval. With `keep_phase`, the grid
// stays where client feedback put it, and the next frame keeps its original slot.
fn claim_next_vsync_slot(
    last_vsync_time: &mut Instant,
    frame_interval: Duration,
    throttle_frames: u32,
    now: Instant,
    keep_phase: bool,
) -> Duration {
    let next_vsync_time = *last_vsync_time + frame_interval * (1 + throttle_frames);

    if next_vsync_time <= now {
        *last_vsync_time = if keep_phase && !frame_interval.is_zero() {
            // The latest slot of the grid that is not after now.
            let behind = (now - next_vsync_time).as_nanos() % frame_interval.as_nanos();
            now - Duration::from_nanos(behind as u64)
        } else {
            now
        };

        Duration::ZERO
    } else {
        *last_vsync_time = next_vsync_time;

        next_vsync_time - now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn us(value: u64) -> Duration {
        Duration::from_micros(value)
    }

    #[test]
    fn frame_pacing_waits_for_an_on_time_frame() {
        let frame_interval = Duration::from_millis(10);
        let start = Instant::now();
        let mut last_vsync_time = start;

        let wait = claim_next_vsync_slot(
            &mut last_vsync_time,
            frame_interval,
            0,
            start + Duration::from_millis(4),
            false,
        );

        assert_eq!(wait, Duration::from_millis(6));
        assert_eq!(last_vsync_time, start + frame_interval);
    }

    #[test]
    fn frame_pacing_does_not_wait_after_missing_a_deadline() {
        let frame_interval = Duration::from_millis(10);
        let start = Instant::now();
        let late_frame_time = start + Duration::from_millis(12);
        let mut last_vsync_time = start;

        let wait = claim_next_vsync_slot(
            &mut last_vsync_time,
            frame_interval,
            0,
            late_frame_time,
            false,
        );

        assert_eq!(wait, Duration::ZERO);
        assert_eq!(last_vsync_time, late_frame_time);
    }

    #[test]
    fn frame_pacing_relocks_after_a_late_frame() {
        let frame_interval = Duration::from_millis(10);
        let start = Instant::now();
        let late_frame_time = start + Duration::from_millis(12);
        let next_frame_time = late_frame_time + Duration::from_millis(4);
        let mut last_vsync_time = start;

        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                late_frame_time,
                false
            ),
            Duration::ZERO
        );
        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                next_frame_time,
                false
            ),
            Duration::from_millis(6)
        );
    }

    #[test]
    fn frame_pacing_does_not_quantize_continuously_late_120_hz_frames() {
        let frame_interval = Duration::from_secs_f32(1.0 / 120.0);
        let start = Instant::now();

        for keep_phase in [false, true] {
            let mut last_vsync_time = start;

            for frame_index in 1..=20 {
                let frame_time = start + Duration::from_millis(9 * frame_index);

                assert_eq!(
                    claim_next_vsync_slot(
                        &mut last_vsync_time,
                        frame_interval,
                        0,
                        frame_time,
                        keep_phase
                    ),
                    Duration::ZERO
                );
                assert!(last_vsync_time <= frame_time);
            }
        }
    }

    #[test]
    fn frame_pacing_keeps_a_stable_deadline_for_early_frames() {
        let frame_interval = Duration::from_millis(10);
        let start = Instant::now();
        let mut last_vsync_time = start;

        for frame_index in 0_u32..5 {
            let expected_deadline = start + frame_interval * (frame_index + 1);
            let frame_time = expected_deadline - Duration::from_millis(4);

            assert_eq!(
                claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, frame_time, false),
                Duration::from_millis(4)
            );
            assert_eq!(last_vsync_time, expected_deadline);
        }
    }

    #[test]
    fn frame_pacing_claims_a_slot_on_every_call() {
        let frame_interval = Duration::from_millis(10);
        let start = Instant::now();
        let mut last_vsync_time = start;

        // Calling twice inside the same interval claims two slots: the second call waits for the
        // slot after the one the first call already took. Callers must not treat this as a query.
        assert_eq!(
            claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, start, false),
            frame_interval
        );
        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                start + Duration::from_millis(1),
                false
            ),
            frame_interval * 2 - Duration::from_millis(1)
        );
    }

    #[test]
    fn frame_pacing_lets_a_frame_through_when_the_deadline_is_now() {
        let frame_interval = Duration::from_millis(10);
        let start = Instant::now();
        let mut last_vsync_time = start;

        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                start + frame_interval,
                false
            ),
            Duration::ZERO
        );
        assert_eq!(last_vsync_time, start + frame_interval);
    }

    #[test]
    fn frame_pacing_never_waits_with_a_zero_interval() {
        // The previous implementation caught up to `now` with a loop adding the frame interval,
        // which spun forever on a zero interval.
        let start = Instant::now();

        for keep_phase in [false, true] {
            let mut last_vsync_time = start;

            for frame_index in 0_u32..3 {
                assert_eq!(
                    claim_next_vsync_slot(
                        &mut last_vsync_time,
                        Duration::ZERO,
                        0,
                        start + Duration::from_millis(frame_index.into()),
                        keep_phase
                    ),
                    Duration::ZERO
                );
            }
        }
    }

    #[test]
    fn frame_pacing_honors_throttle_frames() {
        let frame_interval = Duration::from_millis(10);
        let start = Instant::now();
        let mut last_vsync_time = start;

        let wait = claim_next_vsync_slot(
            &mut last_vsync_time,
            frame_interval,
            1,
            start + Duration::from_millis(4),
            false,
        );

        assert_eq!(wait, Duration::from_millis(16));
        assert_eq!(last_vsync_time, start + frame_interval * 2);
    }

    #[test]
    fn keeping_the_phase_returns_to_the_grid_after_a_late_frame() {
        let frame_interval = ms(10);
        let start = Instant::now();
        let mut last_vsync_time = start;

        // Late by 2 ms: let through at once, but the grid stays on multiples of 10 ms.
        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                start + ms(12),
                true
            ),
            Duration::ZERO
        );
        assert_eq!(last_vsync_time, start + ms(10));
        // The next frame waits for its original slot instead of a slot moved by the late frame.
        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                start + ms(16),
                true
            ),
            ms(4)
        );
        // A frame later than a whole interval skips to the latest slot that has passed.
        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                start + ms(47),
                true
            ),
            Duration::ZERO
        );
        assert_eq!(last_vsync_time, start + ms(40));
    }

    fn locked_pacer(start: Instant, latch_margin_low: f32) -> FramePacer {
        let mut pacer = FramePacer::new_at(ms(10), start);
        pacer.report_client_timing(ms(10), Some(latch_margin_low), Some(ms(1)), start);

        pacer
    }

    #[test]
    fn early_frames_move_the_grid_later_gradually() {
        let start = Instant::now();
        // 5 ms of margin against a 1 ms target: a 0.4 ms correction, at most 0.1 ms per frame.
        let mut pacer = locked_pacer(start, 0.005);

        let waits = (0..5)
            .map(|frame| pacer.claim_next_vsync_slot_at(0, start + ms(10 * frame + 5)))
            .collect::<Vec<_>>();

        assert_eq!(
            waits,
            [
                ms(5) + us(100),
                ms(5) + us(200),
                ms(5) + us(300),
                ms(5) + us(400),
                ms(5) + us(400)
            ]
        );
        assert!(pacer.is_phase_locked(start + ms(45)));
    }

    #[test]
    fn late_frames_move_the_grid_earlier() {
        let start = Instant::now();
        // Frames missed the client latch by 2 ms: move 0.3 ms earlier.
        let mut pacer = locked_pacer(start, -0.002);

        let waits = (0..4)
            .map(|frame| pacer.claim_next_vsync_slot_at(0, start + ms(10 * frame + 5)))
            .collect::<Vec<_>>();

        assert_eq!(
            waits,
            [
                ms(5) - us(100),
                ms(5) - us(200),
                ms(5) - us(300),
                ms(5) - us(300)
            ]
        );
    }

    #[test]
    fn corrections_pause_while_most_frames_miss_their_slot() {
        let start = Instant::now();
        let mut pacer = FramePacer::new_at(ms(10), start);

        // The game is slower than the display: every frame misses its slot.
        for frame in 1..=4 {
            pacer.claim_next_vsync_slot_at(0, start + ms(12 * frame));
        }
        pacer.report_client_timing(ms(10), Some(-0.004), Some(ms(1)), start + ms(48));

        assert_eq!(pacer.pending_phase_shift_ns, 0);
        // The frames are still let through without waiting.
        assert_eq!(
            pacer.claim_next_vsync_slot_at(0, start + ms(60)),
            Duration::ZERO
        );
    }

    #[test]
    fn the_pending_correction_is_bounded_to_half_a_frame() {
        let start = Instant::now();
        let mut pacer = locked_pacer(start, 0.5);

        assert_eq!(pacer.pending_phase_shift_ns, ms(5).as_nanos() as i64);

        pacer.report_client_timing(ms(10), Some(-0.5), Some(ms(1)), start);
        pacer.report_client_timing(ms(10), Some(-0.5), Some(ms(1)), start);
        assert_eq!(pacer.pending_phase_shift_ns, -(ms(5).as_nanos() as i64));
    }

    #[test]
    fn phase_lock_expires_without_feedback() {
        let start = Instant::now();
        let mut pacer = locked_pacer(start, 0.001);
        pacer.claim_next_vsync_slot_at(0, start + ms(5));
        assert!(pacer.is_phase_locked(start + ms(5)));

        let later = start + PHASE_LOCK_TIMEOUT + ms(5);
        assert!(!pacer.is_phase_locked(later));
        // Free running again: a late frame re-anchors the grid.
        assert_eq!(pacer.claim_next_vsync_slot_at(0, later), Duration::ZERO);
        assert_eq!(pacer.last_vsync_time, later);
    }

    #[test]
    fn feedback_without_percentiles_does_not_lock() {
        let start = Instant::now();
        let mut pacer = FramePacer::new_at(ms(10), start);

        pacer.report_client_timing(ms(10), None, Some(ms(1)), start);
        pacer.claim_next_vsync_slot_at(0, start + ms(5));

        assert!(!pacer.is_phase_locked(start + ms(5)));
    }

    #[test]
    fn disabling_phase_lock_restores_free_running_pacing() {
        let start = Instant::now();
        let mut pacer = locked_pacer(start, 0.005);
        pacer.report_client_timing(us(10_050), Some(0.005), Some(ms(1)), start);
        assert_eq!(pacer.frame_interval, us(10_050));

        pacer.report_client_timing(us(10_050), Some(0.005), None, start + ms(1));
        pacer.claim_next_vsync_slot_at(0, start + ms(2));

        assert!(!pacer.is_phase_locked(start + ms(2)));
        assert_eq!(pacer.pending_phase_shift_ns, 0);
        assert_eq!(pacer.frame_interval, ms(10));
    }

    #[test]
    fn the_display_period_only_refines_the_negotiated_interval() {
        let start = Instant::now();
        let mut pacer = FramePacer::new_at(ms(10), start);

        pacer.report_client_timing(us(10_050), Some(0.001), Some(ms(1)), start);
        pacer.claim_next_vsync_slot_at(0, start + ms(1));
        assert_eq!(pacer.frame_interval, us(10_050));
        assert!(pacer.is_phase_locked(start + ms(1)));

        // A different refresh rate is not adopted (SteamVR was told about the negotiated one), and
        // there is no phase to lock to.
        pacer.report_client_timing(ms(12), Some(0.005), Some(ms(1)), start + ms(2));
        assert_eq!(pacer.frame_interval, ms(10));
        assert_eq!(pacer.pending_phase_shift_ns, 0);
        assert!(!pacer.is_phase_locked(start + ms(2)));
    }
}

#[cfg(test)]
mod closed_loop_tests {
    use super::*;
    use std::collections::VecDeque;

    // The same measurement the client makes: time between a frame becoming ready and the deadline
    // of the first refresh after the previously latched frame.
    fn signed_seconds(deadline: Instant, ready: Instant) -> f32 {
        if deadline >= ready {
            (deadline - ready).as_secs_f32()
        } else {
            -(ready - deadline).as_secs_f32()
        }
    }

    pub struct Simulation {
        // (seconds since start, reported 2nd percentile margin)
        pub reported_margins: Vec<(f32, f32)>,
        // Refreshes without a new frame, in seconds since start
        pub repeats: Vec<f32>,
        // (seconds since start, frames waiting at the client after latching)
        pub backlog: Vec<(f32, usize)>,
    }

    // A game that renders in `render` once released, a constant transport latency, and a client
    // display 100 ppm slower than the streamer clock. The client reports every 250 ms like
    // `StatisticsManager::report_frame_latch`, and the report reaches the streamer 2 ms later.
    pub fn simulate(
        seconds: u64,
        render: Duration,
        first_client_deadline: Duration,
        extra_queued_frames: usize,
        target_margin: Option<Duration>,
    ) -> Simulation {
        let start = Instant::now();
        let interval = Duration::from_millis(10);
        let display_period = Duration::from_nanos(10_001_000);
        let transport = Duration::from_millis(20);

        let mut pacer = FramePacer::new_at(interval, start);
        let mut release = start;
        // Extra frames that become ready together with the first one leave a standing backlog.
        let mut available = VecDeque::new();
        for _ in 0..extra_queued_frames {
            available.push_back(start + render + transport);
        }
        let mut next_deadline = start + first_client_deadline;
        let mut first_missed_deadline = None;
        let mut consecutive_repeats = 0;
        let mut latched_frames = 0;
        let mut margins = VecDeque::<(Instant, f32)>::new();
        let mut next_report = start + Duration::from_millis(250);
        let mut reports_in_flight = VecDeque::<(Instant, Option<f32>)>::new();
        let mut simulation = Simulation {
            reported_margins: vec![],
            repeats: vec![],
            backlog: vec![],
        };
        let since_start = |time: Instant| (time - start).as_secs_f32();

        while release < start + Duration::from_secs(seconds) {
            let ready = release + render;
            while reports_in_flight
                .front()
                .is_some_and(|(arrival, _)| *arrival <= ready)
            {
                let (arrival, low) = reports_in_flight.pop_front().unwrap();
                pacer.report_client_timing(display_period, low, target_margin, arrival);
            }
            available.push_back(ready + transport);
            release = ready + pacer.claim_next_vsync_slot_at(0, ready);

            while next_deadline <= release {
                let deadline = next_deadline;
                next_deadline += display_period;

                if available.front().is_some_and(|ready| *ready <= deadline) {
                    let ready = available.pop_front().unwrap();
                    let reference = first_missed_deadline.take().unwrap_or(deadline);
                    if std::mem::take(&mut consecutive_repeats) <= 3 {
                        let bound = 2.0 * display_period.as_secs_f32();
                        let margin = signed_seconds(reference, ready).clamp(-bound, bound);
                        margins.push_back((deadline, margin));
                    }
                    latched_frames += 1;
                    let waiting = available.iter().filter(|ready| **ready <= deadline).count();
                    simulation.backlog.push((since_start(deadline), waiting));
                } else {
                    simulation.repeats.push(since_start(deadline));
                    consecutive_repeats += 1;
                    if latched_frames > 0 {
                        first_missed_deadline.get_or_insert(deadline);
                    }
                }

                while margins
                    .front()
                    .is_some_and(|(time, _)| deadline - *time > Duration::from_secs(1))
                {
                    margins.pop_front();
                }

                if deadline >= next_report {
                    next_report = deadline + Duration::from_millis(250);
                    let low = (margins.len() >= 20).then(|| {
                        let mut sorted = margins.iter().map(|(_, m)| *m).collect::<Vec<_>>();
                        sorted.sort_by(f32::total_cmp);
                        sorted[(sorted.len() as f32 * 0.02) as usize]
                    });
                    if let Some(low) = low {
                        simulation
                            .reported_margins
                            .push((since_start(deadline), low));
                    }
                    reports_in_flight.push_back((deadline + Duration::from_millis(2), low));
                }
            }
        }

        simulation
    }

    fn last_reported_margin(simulation: &Simulation) -> f32 {
        simulation.reported_margins.last().unwrap().1
    }

    fn repeats_after(simulation: &Simulation, seconds: f32) -> usize {
        simulation
            .repeats
            .iter()
            .filter(|time| **time > seconds)
            .count()
    }

    fn max_backlog(simulation: &Simulation, from: f32, to: f32) -> usize {
        simulation
            .backlog
            .iter()
            .filter(|(time, _)| (from..to).contains(time))
            .map(|(_, waiting)| *waiting)
            .max()
            .unwrap()
    }

    #[test]
    fn phase_lock_settles_frames_just_before_the_client_latch() {
        for first_client_deadline_ms in [3, 9] {
            let simulation = simulate(
                30,
                Duration::from_millis(6),
                Duration::from_millis(first_client_deadline_ms),
                0,
                Some(Duration::from_millis(1)),
            );

            let margin = last_reported_margin(&simulation);
            assert!((margin - 0.001).abs() < 0.0001, "margin {margin}");
            // Only the refreshes before the first frame arrived.
            assert_eq!(repeats_after(&simulation, 0.5), 0);
        }
    }

    #[test]
    fn phase_lock_drains_a_standing_backlog_without_skipping_or_repeating() {
        let run = |target_margin| {
            simulate(
                30,
                Duration::from_millis(6),
                Duration::from_millis(3),
                1,
                target_margin,
            )
        };

        // Free running, a frame that is queued behind another keeps waiting an extra refresh.
        let free_running = run(None);
        assert_eq!(max_backlog(&free_running, 25.0, 30.0), 1);

        let locked = run(Some(Duration::from_millis(1)));
        assert_eq!(max_backlog(&locked, 0.0, 1.0), 1);
        assert_eq!(max_backlog(&locked, 15.0, 30.0), 0);
        // Every frame is still shown once: the backlogged frame covers the refresh the delayed
        // frames no longer reach.
        assert_eq!(repeats_after(&locked, 0.5), 0);
        assert!((last_reported_margin(&locked) - 0.001).abs() < 0.0001);
    }
}

// The closed loop above latches on a fixed display grid. This one also models what moves the
// latch deadline of the client: its frame loop. xrWaitFrame() wakes the loop on the display grid,
// the loop waits a bounded time for a decoded frame, renders, and has to call xrWaitFrame() again
// before the next wake-up. The compositor shows the newest submitted frame once per refresh.
#[cfg(test)]
mod client_loop_tests {
    use super::*;
    use std::collections::VecDeque;

    const INTERVAL: Duration = Duration::from_nanos(8_333_333);
    // The headset clock runs 100 ppm slower than the streamer clock.
    const DISPLAY_PERIOD: Duration = Duration::from_nanos(8_334_166);
    const GAME_RENDER: Duration = Duration::from_millis(4);
    const TRANSPORT: Duration = Duration::from_millis(20);
    const MAX_JITTER: Duration = Duration::from_micros(600);
    const TARGET_MARGIN: Duration = Duration::from_millis(1);
    const SETTLING_TIME: f32 = 20.0;

    // What xrWaitFrame() does when it is called after the wake-up time it would have returned at.
    #[derive(Clone, Copy, Debug)]
    enum LateWakeUp {
        // Returns at once: the loop keeps running, shifted against the display.
        AtOnce,
        // Returns at the following wake-up: the loop stays on the grid and loses a refresh.
        NextRefresh,
    }

    struct Run {
        // (seconds since start, reported 2nd percentile margin)
        reported_margins: Vec<(f32, f32)>,
        // Refreshes that showed the previous frame again, in seconds since start
        display_repeats: Vec<f32>,
        // Refreshes that received more than one frame, of which only the newest is shown
        display_skips: Vec<f32>,
        // Iterations that called xrWaitFrame() after its wake-up time
        late_iterations: Vec<f32>,
    }

    impl Run {
        fn settled(times: &[f32]) -> usize {
            times.iter().filter(|time| **time > SETTLING_TIME).count()
        }

        fn settled_margins(&self) -> impl Iterator<Item = f32> + '_ {
            self.reported_margins
                .iter()
                .filter(|(time, _)| *time > SETTLING_TIME)
                .map(|(_, margin)| *margin)
        }
    }

    struct Jitter(u64);

    impl Jitter {
        fn next(&mut self) -> Duration {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);

            MAX_JITTER.mul_f64((self.0 >> 33) as f64 / (1_u64 << 31) as f64)
        }
    }

    fn signed_seconds(deadline: Instant, ready: Instant) -> f32 {
        if deadline >= ready {
            (deadline - ready).as_secs_f32()
        } else {
            -(ready - deadline).as_secs_f32()
        }
    }

    // `max_wait` is the longest the client waits for a decoded frame, `render` the time from the
    // end of that wait to its next xrWaitFrame() call. Game and transport times both jitter.
    fn simulate(late_wake_up: LateWakeUp, max_wait: Duration, render: Duration) -> Run {
        let start = Instant::now();
        let end = start + Duration::from_secs(60);
        let since_start = |time: Instant| (time - start).as_secs_f32();
        let mut jitter = Jitter(1);

        // Streamer
        let mut pacer = FramePacer::new_at(INTERVAL, start);
        let mut release = start;
        let mut reports_in_flight = VecDeque::<(Instant, Option<f32>)>::new();
        let mut frames_in_flight = VecDeque::<(Instant, u64)>::new();
        let mut next_frame = 0;

        // Client
        let wake_up = |slot: u32| start + Duration::from_millis(3) + DISPLAY_PERIOD * slot;
        let mut slot = 0;
        let mut wait_frame_call = start;
        let mut decoded = VecDeque::<(Instant, u64)>::new();
        let mut first_missed_deadline = None;
        let mut consecutive_repeats = 0;
        let mut latched_frames = 0;
        let mut margins = VecDeque::<(Instant, f32)>::new();
        let mut next_report = start + Duration::from_millis(250);
        // (xrEndFrame() time, frame)
        let mut submissions = Vec::<(Instant, u64)>::new();

        let mut run = Run {
            reported_margins: vec![],
            display_repeats: vec![],
            display_skips: vec![],
            late_iterations: vec![],
        };

        loop {
            // Where the next client iteration starts. Wake-ups it is too late for are lost.
            let mut next_slot = slot;
            while wake_up(next_slot + 1) <= wait_frame_call {
                next_slot += 1;
            }
            let late = wait_frame_call > wake_up(next_slot);
            if late && matches!(late_wake_up, LateWakeUp::NextRefresh) {
                next_slot += 1;
            }
            let iteration_start = wait_frame_call.max(wake_up(next_slot));

            // The earlier of the next streamer frame and the next client iteration.
            if release + GAME_RENDER <= iteration_start {
                let ready = release + GAME_RENDER + jitter.next();
                if ready >= end {
                    break;
                }
                while reports_in_flight
                    .front()
                    .is_some_and(|(arrival, _)| *arrival <= ready)
                {
                    let (arrival, low) = reports_in_flight.pop_front().unwrap();
                    pacer.report_client_timing(DISPLAY_PERIOD, low, Some(TARGET_MARGIN), arrival);
                }
                frames_in_flight.push_back((ready + TRANSPORT + jitter.next(), next_frame));
                next_frame += 1;
                release = ready + pacer.claim_next_vsync_slot_at(0, ready);

                continue;
            }

            if late {
                run.late_iterations.push(since_start(iteration_start));
            }
            slot = next_slot + 1;

            let deadline = iteration_start + max_wait;
            while frames_in_flight
                .front()
                .is_some_and(|(arrival, _)| *arrival <= deadline)
            {
                decoded.push_back(frames_in_flight.pop_front().unwrap());
            }

            let wait_end = if let Some((ready, frame)) = decoded.pop_front() {
                let wait_end = ready.max(iteration_start);
                let reference = first_missed_deadline.take().unwrap_or(deadline);
                if std::mem::take(&mut consecutive_repeats) <= 3 {
                    let bound = 2.0 * DISPLAY_PERIOD.as_secs_f32();
                    let margin = signed_seconds(reference, ready).clamp(-bound, bound);
                    margins.push_back((wait_end, margin));
                }
                latched_frames += 1;
                submissions.push((wait_end + render, frame));

                wait_end
            } else {
                consecutive_repeats += 1;
                if latched_frames > 0 {
                    first_missed_deadline.get_or_insert(deadline);
                }

                deadline
            };

            while margins
                .front()
                .is_some_and(|(time, _)| wait_end - *time > Duration::from_secs(1))
            {
                margins.pop_front();
            }
            if wait_end >= next_report {
                next_report = wait_end + Duration::from_millis(250);
                let low = (margins.len() >= 20).then(|| {
                    let mut sorted = margins.iter().map(|(_, m)| *m).collect::<Vec<_>>();
                    sorted.sort_by(f32::total_cmp);
                    sorted[(sorted.len() as f32 * 0.02) as usize]
                });
                if let Some(low) = low {
                    run.reported_margins.push((since_start(wait_end), low));
                }
                reports_in_flight.push_back((wait_end + Duration::from_millis(2), low));
            }

            wait_frame_call = wait_end + render;
        }

        // The compositor takes the newest frame submitted by each wake-up time.
        let mut submissions = submissions.into_iter().peekable();
        let mut showing_a_frame = false;
        for slot in 1.. {
            let latch = wake_up(slot);
            if latch >= end {
                break;
            }

            let mut new_frames = 0;
            while submissions.next_if(|(time, _)| *time <= latch).is_some() {
                new_frames += 1;
            }
            match new_frames {
                0 if showing_a_frame => run.display_repeats.push(since_start(latch)),
                0 => (),
                1 => showing_a_frame = true,
                _ => run.display_skips.push(since_start(latch)),
            }
        }

        run
    }

    #[test]
    fn phase_lock_settles_when_the_client_frame_loop_fits_in_the_display_period() {
        for late_wake_up in [LateWakeUp::AtOnce, LateWakeUp::NextRefresh] {
            // 3 ms to render, and a wait that leaves 4 ms of the period for it.
            let run = simulate(
                late_wake_up,
                DISPLAY_PERIOD - Duration::from_millis(4),
                Duration::from_millis(3),
            );

            for margin in run.settled_margins() {
                assert!(
                    (margin - TARGET_MARGIN.as_secs_f32()).abs() < 0.0003,
                    "{late_wake_up:?}: margin {margin}"
                );
            }
            assert!(run.settled_margins().count() > 100);
            assert_eq!(Run::settled(&run.late_iterations), 0, "{late_wake_up:?}");
            assert_eq!(Run::settled(&run.display_repeats), 0, "{late_wake_up:?}");
            assert_eq!(Run::settled(&run.display_skips), 0, "{late_wake_up:?}");
        }
    }

    // Why the client must keep its render time inside the display period: if it waits a whole
    // period and then renders, it calls xrWaitFrame() late. Its next deadline then follows the
    // previous frame instead of the display, so the margin it reports no longer depends on the
    // phase of the grid. The grid is steered without ever reaching the target, and frames keep
    // crossing the compositor latch.
    #[test]
    fn a_client_frame_loop_longer_than_the_display_period_cannot_be_locked() {
        for late_wake_up in [LateWakeUp::AtOnce, LateWakeUp::NextRefresh] {
            let run = simulate(late_wake_up, DISPLAY_PERIOD, Duration::from_millis(3));

            for margin in run.settled_margins() {
                assert!(
                    margin > TARGET_MARGIN.as_secs_f32() + 0.0005,
                    "{late_wake_up:?}: margin {margin}"
                );
            }
            assert!(
                Run::settled(&run.display_repeats) > 5,
                "{late_wake_up:?}: {} repeats",
                Run::settled(&run.display_repeats)
            );
        }
    }
}
