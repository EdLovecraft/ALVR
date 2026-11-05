use super::{VideoDecoderConfig, frame_queue::FrameQueue};
use crate::thread_priority::{self, CriticalThreadKind};
use alvr_common::{
    RelaxedAtomic, ToAny,
    anyhow::{Context, Result, anyhow, bail},
    error, info,
    parking_lot::{Condvar, Mutex},
    warn,
};
use alvr_session::{CodecType, MediacodecPropType};
use ndk::{
    hardware_buffer::HardwareBufferUsage,
    media::{
        image_reader::{AcquireResult, Image, ImageFormat, ImageReader},
        media_codec::{
            DequeuedInputBufferResult, DequeuedOutputBufferInfoResult, MediaCodec,
            MediaCodecDirection, MediaFormat,
        },
    },
};
use std::{
    cell::Cell,
    ffi::c_void,
    ops::Deref,
    ptr,
    sync::{
        Arc, Weak,
        atomic::{AtomicU32, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

thread_local! {
    // The ImageReader invokes its listener on a thread it owns. Declare it once, on first use.
    static LISTENER_THREAD_DECLARED: Cell<bool> = const { Cell::new(false) };
}

struct FakeThreadSafe<T>(T);
unsafe impl<T> Send for FakeThreadSafe<T> {}
unsafe impl<T> Sync for FakeThreadSafe<T> {}

impl<T> Deref for FakeThreadSafe<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

type SharedMediaCodec = Arc<FakeThreadSafe<MediaCodec>>;

// Allow brief input-buffer contention without dropping a reference frame and waiting for an IDR.
// Keep the budget well below a refresh period: this runs on the video receive thread. MediaCodec
// returns immediately when a buffer is ready; this is a timeout, not a delay on every frame.
const INPUT_BUFFER_TIMEOUT: Duration = Duration::from_millis(1);

pub struct VideoDecoderSink {
    inner: Arc<Mutex<Option<SharedMediaCodec>>>,
}

unsafe impl Send for VideoDecoderSink {}

impl VideoDecoderSink {
    // Wait briefly for an input buffer, then submit the frame. Returns false if the wait times out.
    pub fn push_frame_nal(&mut self, timestamp: Duration, data: &[u8]) -> Result<bool> {
        let Some(decoder) = &*self.inner.lock() else {
            // This might happen only during destruction
            return Ok(false);
        };

        match decoder.dequeue_input_buffer(INPUT_BUFFER_TIMEOUT) {
            Ok(DequeuedInputBufferResult::Buffer(mut buffer)) => {
                unsafe {
                    ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        buffer.buffer_mut().as_mut_ptr().cast(),
                        data.len(),
                    )
                };

                // NB: the function expects the timestamp in micros, but nanos is used to have
                // complete precision, so when converted back to Duration it can compare correctly
                // to other Durations
                decoder.queue_input_buffer(buffer, 0, data.len(), timestamp.as_nanos() as _, 0)?;

                Ok(true)
            }
            Ok(DequeuedInputBufferResult::TryAgainLater) => Ok(false),
            Err(e) => bail!("{e}"),
        }
    }
}

const MAX_ACQUIRED_IMAGES: usize = 10;

struct QueuedImage {
    timestamp: Duration,
    image: Image,
}
unsafe impl Send for QueuedImage {}

type SharedImageQueue = Arc<Mutex<FrameQueue<QueuedImage>>>;

// Access the image queue synchronously.
pub struct VideoDecoderSource {
    running: Arc<RelaxedAtomic>,
    // Keep the ImageReader in the thread result even after a decoder error: it must outlive every
    // acquired Image, including the one currently held by the consumer.
    dequeue_thread: Option<JoinHandle<Option<FakeThreadSafe<ImageReader>>>>,
    image_queue: SharedImageQueue,
    frame_available: Arc<Condvar>,
    current_image: Option<QueuedImage>,
    // Decoded frames discarded before display, by queue overflow or backlog recovery.
    dropped_frames: Arc<AtomicU32>,
}

unsafe impl Send for VideoDecoderSource {}

impl VideoDecoderSource {
    // If no frame is ready, wait up to `timeout` for the decoder to produce one. The application
    // MUST finish using the returned buffer before calling this function again.
    pub fn dequeue_frame_timeout(&mut self, timeout: Duration) -> Option<(Duration, *mut c_void)> {
        // The previous buffer can now be reused by the decoder. Release it without the queue lock.
        drop(self.current_image.take());

        let (next_image, discarded) = {
            let mut image_queue_lock = self.image_queue.lock();

            if image_queue_lock.is_empty() {
                // Submit a layer at every vsync even when the decoder is late. A timeout is normal.
                self.frame_available.wait_while_for(
                    &mut image_queue_lock,
                    |image_queue| image_queue.is_empty(),
                    timeout,
                );
            }

            image_queue_lock.dequeue()
        };
        if discarded.is_some() {
            self.dropped_frames.fetch_add(1, Ordering::Relaxed);
        }
        drop(discarded);

        // Once dequeued, only this source can release the image. Producer-side overflow and error
        // recovery operate exclusively on pending frames, never on a buffer being rendered.
        self.current_image = next_image;
        self.current_image.as_ref().map(|queued_image| {
            (
                queued_image.timestamp,
                queued_image
                    .image
                    .hardware_buffer()
                    .unwrap()
                    .as_ptr()
                    .cast(),
            )
        })
    }

    // Returns the number of decoded frames discarded since the previous call.
    pub fn take_dropped_frames(&self) -> u32 {
        self.dropped_frames.swap(0, Ordering::Relaxed)
    }
}

impl Drop for VideoDecoderSource {
    fn drop(&mut self) {
        self.running.set(false);

        let image_reader = self
            .dequeue_thread
            .take()
            .and_then(|thread| thread.join().ok())
            .flatten();

        // Images must be released before their parent ImageReader, and outside the queue lock.
        drop(self.current_image.take());
        let pending = self.image_queue.lock().clear();
        drop(pending);
        drop(image_reader);
    }
}

fn mime_for_codec(codec: CodecType) -> &'static str {
    match codec {
        CodecType::H264 => "video/avc",
        CodecType::Hevc => "video/hevc",
        CodecType::AV1 => "video/av01",
    }
}

// Attempts to create a MediaCodec, and then configure and start it.
fn decoder_setup(
    codec_type: CodecType,
    is_software: bool,
    format: &MediaFormat,
    image_reader: &ImageReader,
) -> Result<MediaCodec> {
    let decoder = if is_software {
        let sw_codec_name = match codec_type {
            CodecType::H264 => "OMX.google.h264.decoder",
            CodecType::Hevc => "OMX.google.hevc.decoder",
            CodecType::AV1 => bail!("AV1 is not supported for software decoding"),
        };
        MediaCodec::from_codec_name(&sw_codec_name)
            .ok_or(anyhow!("no such codec: {}", &sw_codec_name))?
    } else {
        let mime = mime_for_codec(codec_type);
        MediaCodec::from_decoder_type(&mime)
            .ok_or(anyhow!("unable to find decoder for mime type: {}", &mime))?
    };
    decoder
        .configure(
            &format,
            Some(&image_reader.window()?),
            MediaCodecDirection::Decoder,
        )
        .with_context(|| format!("failed to configure decoder"))?;

    decoder
        .start()
        .with_context(|| format!("failed to start decoder"))?;

    Ok(decoder)
}

// The ImageReader can outlive this thread after an error. Its listener must not keep the client
// context alive once the decoder's strong callback reference is gone.
fn decoder_lifecycle(
    config: VideoDecoderConfig,
    csd_0: Vec<u8>,
    frame_result_callback: Weak<impl Fn(Result<Duration>) + Send + Sync + 'static>,
    running: Arc<RelaxedAtomic>,
    decoder_sink: Arc<Mutex<Option<SharedMediaCodec>>>,
    decoder_ready_notifier: Arc<Condvar>,
    image_queue: SharedImageQueue,
    frame_available: Arc<Condvar>,
    dropped_frames: Arc<AtomicU32>,
    image_reader: &mut ImageReader,
) -> Result<()> {
    image_reader.set_image_listener(Box::new({
        let image_queue = Arc::clone(&image_queue);
        let frame_available = Arc::clone(&frame_available);
        let running = Arc::clone(&running);
        move |image_reader| {
            if !running.value() {
                return;
            }

            if !LISTENER_THREAD_DECLARED.replace(true) {
                thread_priority::declare_current_thread(CriticalThreadKind::Decoder);
            }

            let overflow = image_queue.lock().discard_oldest_if_full();
            if overflow.is_some() {
                dropped_frames.fetch_add(1, Ordering::Relaxed);
                warn!("Video frame queue overflow!");
            }
            drop(overflow);

            // Acquiring an image may wait for a fence. Neither it nor the statistics callback may
            // hold the queue lock and delay the compositor's bounded dequeue.
            match image_reader.acquire_next_image() {
                Ok(AcquireResult::Image(image)) => {
                    let timestamp = Duration::from_nanos(image.timestamp().unwrap() as u64);

                    // Record decode completion before making the image visible to the consumer.
                    if let Some(callback) = frame_result_callback.upgrade() {
                        callback(Ok(timestamp));
                    }

                    let queued_image = QueuedImage { timestamp, image };
                    let discarded = {
                        let mut queue = image_queue.lock();
                        // Check under the lock so teardown cannot clear the queue only for a late
                        // callback to repopulate it before the ImageReader is destroyed.
                        if running.value() {
                            let discarded = queue.push(queued_image);
                            if discarded.is_some() {
                                dropped_frames.fetch_add(1, Ordering::Relaxed);
                            }

                            discarded
                        } else {
                            Some(queued_image)
                        }
                    };
                    drop(discarded);
                    frame_available.notify_one();
                }
                Ok(AcquireResult::NoBufferAvailable) => (),
                Ok(e) => {
                    error!("ImageReader error: {e:?}");

                    let pending = image_queue.lock().clear();
                    drop(pending);
                }
                Err(e) => {
                    error!("ImageReader error: {e}");

                    let pending = image_queue.lock().clear();
                    drop(pending);
                }
            }
        }
    }))?;

    // Keep the buffer removal listener registered for the lifetime of the ImageReader.
    image_reader.set_buffer_removed_listener(Box::new(|_, _| ()))?;

    let mime = mime_for_codec(config.codec);

    let mut format = MediaFormat::new();
    format.set_str("mime", mime);
    // Given https://github.com/alvr-org/ALVR/pull/1933#discussion_r1431902906 - change at own risk.
    // It might be harmless, it might not be, but it's definitely a risk.
    format.set_i32("width", 512);
    format.set_i32("height", 1024);
    format.set_buffer("csd-0", &csd_0);

    for (key, prop) in &config.options {
        let maybe_error = match prop.ty {
            MediacodecPropType::Float => prop
                .value
                .parse()
                .map(|value| format.set_f32(key, value))
                .to_any(),
            MediacodecPropType::Int32 => prop
                .value
                .parse()
                .map(|value| format.set_i32(key, value))
                .to_any(),
            MediacodecPropType::Int64 => prop
                .value
                .parse()
                .map(|value| format.set_i64(key, value))
                .to_any(),
            MediacodecPropType::String => Ok(format.set_str(key, &prop.value)),
        };

        if let Err(e) = maybe_error {
            error!("Failed to set property {key} to {}: {e}", prop.value);
        }
    }

    info!("Using AMediaCodec format:{} ", format);

    let decoder = if config.force_software_decoder {
        decoder_setup(config.codec, true, &format, &image_reader)?
    } else {
        // Hardware decoders sometimes fail at the CSD-0.
        // May as well fall back if this occurs.
        match decoder_setup(config.codec, false, &format, &image_reader) {
            Ok(d) => d,
            Err(e) => {
                // would be "warn!" but this is a severe caveat and a pretty major error.
                error!("Attempting software fallback due to error in default decoder: {e:#}");

                decoder_setup(config.codec, true, &format, &image_reader)?
            }
        }
    };

    let decoder = Arc::new(FakeThreadSafe(decoder));

    {
        let mut decoder_lock = decoder_sink.lock();

        *decoder_lock = Some(Arc::clone(&decoder));

        decoder_ready_notifier.notify_one();
    }

    let mut error_counter = 0;
    while running.value() {
        match decoder.dequeue_output_buffer(Duration::from_millis(5)) {
            Ok(DequeuedOutputBufferInfoResult::Buffer(buffer)) => {
                // queue_input_buffer stores nanoseconds in MediaCodec's microsecond timestamp field.
                // Pass that value explicitly so ImageReader preserves the exact frame timestamp.
                // release_output_buffer(buffer, true) would multiply it by 1000, preventing the
                // compositor from matching the frame to its eye poses, FoVs and foveation centers.
                let presentation_time_ns = buffer.info().presentation_time_us();

                if let Err(e) = decoder.release_output_buffer_at_time(buffer, presentation_time_ns)
                {
                    error!("Decoder dequeue error: {e}");
                }
            }
            Ok(DequeuedOutputBufferInfoResult::TryAgainLater) => continue,
            Ok(i) => info!("Decoder dequeue event: {i:?}"),
            Err(e) => {
                error!("Decoder dequeue error: {e}");

                error_counter += 1;
                if error_counter > 10 {
                    bail!("Too many decoder errors: {e}");
                }

                // lessen logcat flood (just in case)
                thread::sleep(Duration::from_millis(50));

                continue;
            }
        }

        error_counter = 0;
    }

    // Destroy all resources
    decoder_sink.lock().take(); // Make sure the shared ref is deleted first
    decoder.stop()?;
    drop(decoder);

    Ok(())
}

// Create a sink/source pair
pub fn video_decoder_split(
    config: VideoDecoderConfig,
    csd_0: Vec<u8>,
    frame_result_callback: impl Fn(Result<Duration>) + Send + Sync + 'static,
) -> Result<(VideoDecoderSink, VideoDecoderSource)> {
    let running = Arc::new(RelaxedAtomic::new(true));
    let decoder_sink = Arc::new(Mutex::new(None::<SharedMediaCodec>));
    let decoder_ready_notifier = Arc::new(Condvar::new());
    // Leave room for the consumer's current image and an image being acquired by the callback.
    let image_queue = Arc::new(Mutex::new(FrameQueue::new(
        config.max_buffering_frames,
        config.buffering_history_weight,
        MAX_ACQUIRED_IMAGES - 2,
    )));
    let frame_available = Arc::new(Condvar::new());
    let dropped_frames = Arc::new(AtomicU32::new(0));

    let dequeue_thread = thread::spawn({
        let config = config.clone();
        let running = Arc::clone(&running);
        let decoder_sink = Arc::clone(&decoder_sink);
        let decoder_ready_notifier = Arc::clone(&decoder_ready_notifier);
        let image_queue = Arc::clone(&image_queue);
        let frame_available = Arc::clone(&frame_available);
        let dropped_frames = Arc::clone(&dropped_frames);
        move || {
            thread_priority::declare_current_thread(CriticalThreadKind::Decoder);

            let mut image_reader = match ImageReader::new_with_usage(
                1,
                1,
                ImageFormat::PRIVATE,
                HardwareBufferUsage::GPU_SAMPLED_IMAGE,
                MAX_ACQUIRED_IMAGES as i32,
            ) {
                Ok(reader) => reader,
                Err(e) => {
                    frame_result_callback(Err(anyhow!("{e}")));
                    return None;
                }
            };

            let frame_result_callback = Arc::new(frame_result_callback);

            if let Err(e) = decoder_lifecycle(
                config,
                csd_0,
                Arc::downgrade(&frame_result_callback),
                Arc::clone(&running),
                decoder_sink,
                decoder_ready_notifier,
                Arc::clone(&image_queue),
                frame_available,
                dropped_frames,
                &mut image_reader,
            ) {
                frame_result_callback(Err(e));
            }

            running.set(false);
            let pending = image_queue.lock().clear();
            drop(pending);

            Some(FakeThreadSafe(image_reader))
        }
    });

    // Make sure the decoder is ready: we don't want to try to enqueue frame and lose them, to avoid
    // image corruption.
    {
        let mut decoder_lock = decoder_sink.lock();

        if decoder_lock.is_none() {
            // No spurious wakeups
            decoder_ready_notifier.wait(&mut decoder_lock);
        }
    }

    let sink = VideoDecoderSink {
        inner: decoder_sink,
    };
    let source = VideoDecoderSource {
        running,
        dequeue_thread: Some(dequeue_thread),
        image_queue,
        frame_available,
        current_image: None,
        dropped_frames,
    };

    Ok((sink, source))
}
