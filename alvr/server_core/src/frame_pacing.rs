use std::time::{Duration, Instant};

// This state has its own lock so statistics processing cannot delay the pacing deadline.
pub struct FramePacer {
    last_vsync_time: Instant,
    frame_interval: Duration,
}

impl FramePacer {
    pub fn new(frame_interval: Duration) -> Self {
        Self {
            last_vsync_time: Instant::now(),
            frame_interval,
        }
    }

    // NB: this call is non-blocking, waiting should be done externally
    pub fn claim_next_vsync_slot(&mut self, throttle_frames: u32) -> Duration {
        claim_next_vsync_slot(
            &mut self.last_vsync_time,
            self.frame_interval,
            throttle_frames,
            Instant::now(),
        )
    }
}

// Claims the next vsync slot and returns how long the caller must wait for it. This is not a pure
// query: every call advances the pacing phase by one frame interval (plus any extra frames SteamVR
// asked to throttle by), so it must be called exactly once per produced frame, and only while
// server frame pacing is enabled. A frame that arrives after its slot has passed is let through
// immediately and re-anchors the phase, which keeps framerates below the refresh rate from being
// quantized to an integer divisor of it (a slightly late 90 Hz frame would otherwise be delayed to
// the 45 Hz slot).
fn claim_next_vsync_slot(
    last_vsync_time: &mut Instant,
    frame_interval: Duration,
    throttle_frames: u32,
    now: Instant,
) -> Duration {
    let next_vsync_time = *last_vsync_time + frame_interval * (1 + throttle_frames);

    if next_vsync_time <= now {
        *last_vsync_time = now;

        Duration::ZERO
    } else {
        *last_vsync_time = next_vsync_time;

        next_vsync_time - now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let wait = claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, late_frame_time);

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
            claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, late_frame_time,),
            Duration::ZERO
        );
        assert_eq!(
            claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, next_frame_time,),
            Duration::from_millis(6)
        );
    }

    #[test]
    fn frame_pacing_does_not_quantize_continuously_late_120_hz_frames() {
        let frame_interval = Duration::from_secs_f32(1.0 / 120.0);
        let start = Instant::now();
        let mut last_vsync_time = start;

        for frame_index in 1..=5 {
            let frame_time = start + Duration::from_millis(9 * frame_index);

            assert_eq!(
                claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, frame_time,),
                Duration::ZERO
            );
            assert_eq!(last_vsync_time, frame_time);
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
                claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, frame_time,),
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
            claim_next_vsync_slot(&mut last_vsync_time, frame_interval, 0, start),
            frame_interval
        );
        assert_eq!(
            claim_next_vsync_slot(
                &mut last_vsync_time,
                frame_interval,
                0,
                start + Duration::from_millis(1)
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
                start + frame_interval
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
        let mut last_vsync_time = start;

        for frame_index in 0_u32..3 {
            assert_eq!(
                claim_next_vsync_slot(
                    &mut last_vsync_time,
                    Duration::ZERO,
                    0,
                    start + Duration::from_millis(frame_index.into())
                ),
                Duration::ZERO
            );
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
        );

        assert_eq!(wait, Duration::from_millis(16));
        assert_eq!(last_vsync_time, start + frame_interval * 2);
    }
}
