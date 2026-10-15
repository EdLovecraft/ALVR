use std::{collections::VecDeque, mem};

// Only pending frames live here. A dequeued frame is owned by the consumer, so neither overflow
// recovery nor clearing this queue can release a buffer that is still being rendered.
pub(super) struct FrameQueue<T> {
    frames: VecDeque<T>,
    capacity: usize,
    max_buffering_frames: f32,
    buffering_history_weight: f32,
    buffering_running_average: f32,
}

impl<T> FrameQueue<T> {
    pub fn new(
        max_buffering_frames: f32,
        buffering_history_weight: f32,
        max_pending_frames: usize,
    ) -> Self {
        Self {
            frames: VecDeque::new(),
            capacity: ((2. * max_buffering_frames).ceil() as usize).clamp(1, max_pending_frames),
            max_buffering_frames,
            buffering_history_weight,
            buffering_running_average: 0.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    // Return discarded frames to the caller so potentially expensive buffer destruction can happen
    // after releasing the queue lock. Make room before acquiring another ImageReader buffer.
    pub fn discard_oldest_if_full(&mut self) -> Option<T> {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front()
        } else {
            None
        }
    }

    pub fn push(&mut self, frame: T) -> Option<T> {
        let discarded = self.discard_oldest_if_full();
        self.frames.push_back(frame);
        discarded
    }

    // Returns (next frame, discarded frame). Only frames behind the next one count as backlog,
    // and a stale average must never discard the only frame currently available for display.
    pub fn dequeue(&mut self) -> (Option<T>, Option<T>) {
        let buffered_frames = self.frames.len().saturating_sub(1);
        self.buffering_running_average = self.buffering_running_average
            * self.buffering_history_weight
            + buffered_frames as f32 * (1. - self.buffering_history_weight);

        let discarded = if self.frames.len() > 1
            && self.buffering_running_average > self.max_buffering_frames
        {
            self.frames.pop_front()
        } else {
            None
        };

        (self.frames.pop_front(), discarded)
    }

    pub fn clear(&mut self) -> VecDeque<T> {
        self.buffering_running_average = 0.0;
        mem::take(&mut self.frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn stale_average_never_discards_the_only_frame() {
        let mut queue = FrameQueue::new(2.0, 0.90, 8);
        queue.buffering_running_average = 3. * (1. - 0.90_f32.powi(20));

        for frame in 0..3 {
            assert!(queue.push(frame).is_none());
        }
        assert_eq!(queue.dequeue(), (Some(1), Some(0)));
        assert!(queue.push(3).is_none());
        assert_eq!(queue.dequeue(), (Some(3), Some(2)));
        assert!(queue.push(4).is_none());
        assert_eq!(queue.dequeue(), (Some(4), None));
        assert!(queue.buffering_running_average > 2.0);
    }

    #[test]
    fn a_single_on_time_frame_does_not_count_as_backlog() {
        let mut queue = FrameQueue::new(1.0, 0.90, 8);
        for frame in 0..100 {
            assert!(queue.push(frame).is_none());
            assert_eq!(queue.dequeue(), (Some(frame), None));
            assert_eq!(queue.buffering_running_average, 0.0);
        }
    }

    #[test]
    fn empty_queue_does_not_produce_or_discard_a_frame() {
        let mut queue = FrameQueue::<u32>::new(2.0, 0.90, 8);
        queue.buffering_running_average = 3.0;
        assert_eq!(queue.dequeue(), (None, None));
    }

    #[test]
    fn on_time_frames_keep_fifo_order() {
        let mut queue = FrameQueue::new(2.0, 0.90, 8);
        for frame in 0..3 {
            assert!(queue.push(frame).is_none());
        }
        for frame in 0..3 {
            assert_eq!(queue.dequeue(), (Some(frame), None));
        }
    }

    #[test]
    fn capacity_reserves_space_for_consumer_and_image_acquisition() {
        let mut queue = FrameQueue::new(10.0, 0.90, 8);
        assert_eq!(queue.capacity, 8);
        for frame in 0..8 {
            assert!(queue.push(frame).is_none());
        }
        assert_eq!(queue.discard_oldest_if_full(), Some(0));
        assert!(queue.push(8).is_none());
        assert_eq!(queue.push(9), Some(1));
        assert_eq!(queue.frames.len(), 8);
    }

    struct TrackedFrame {
        id: usize,
        released: Arc<Mutex<Vec<usize>>>,
    }

    impl Drop for TrackedFrame {
        fn drop(&mut self) {
            self.released.lock().unwrap().push(self.id);
        }
    }

    #[test]
    fn overflow_and_clear_cannot_release_a_dequeued_frame() {
        let released = Arc::new(Mutex::new(Vec::new()));
        let mut queue = FrameQueue::new(1.0, 0.90, 8);
        let frame = |id| TrackedFrame {
            id,
            released: Arc::clone(&released),
        };

        assert!(queue.push(frame(0)).is_none());
        let (current, discarded) = queue.dequeue();
        assert!(discarded.is_none());
        assert!(queue.push(frame(1)).is_none());
        assert!(queue.push(frame(2)).is_none());
        let overflow = queue.push(frame(3));
        assert!(released.lock().unwrap().is_empty());
        drop(overflow);
        assert_eq!(*released.lock().unwrap(), [1]);

        let pending = queue.clear();
        assert!(queue.is_empty());
        assert_eq!(*released.lock().unwrap(), [1]);
        drop(pending);
        assert_eq!(*released.lock().unwrap(), [1, 2, 3]);
        drop(current);
        assert_eq!(*released.lock().unwrap(), [1, 2, 3, 0]);
    }

    #[test]
    fn clearing_pending_frames_resets_the_backlog_average() {
        let mut queue = FrameQueue::new(2.0, 0.90, 8);
        queue.buffering_running_average = 3.0;
        assert!(queue.push(0).is_none());
        assert_eq!(queue.clear(), VecDeque::from([0]));
        assert_eq!(queue.buffering_running_average, 0.0);
    }
}
