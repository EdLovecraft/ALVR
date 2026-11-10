use crate::{
    graphics::{self, ProjectionLayerAlphaConfig, ProjectionLayerBuilder},
    interaction::{self, InteractionContext, InteractionSourcesConfig},
};
use alvr_client_core::{
    ClientCoreContext, FrameLatch, VideoFrameMetadata,
    video_decoder::{self, VideoDecoderConfig, VideoDecoderSource},
};
use alvr_common::{
    AlvrFoveatedEncodingParams, DETACHED_CONTROLLER_LEFT_ID, DETACHED_CONTROLLER_RIGHT_ID,
    HAND_LEFT_ID, HAND_RIGHT_ID, HEAD_ID, Pose, RelaxedAtomic, ViewParams,
    anyhow::Result,
    error,
    glam::{UVec2, Vec2},
    parking_lot::RwLock,
};
use alvr_graphics::{GraphicsContext, StreamRenderer, StreamViewParams};
use alvr_packets::{ClientStreamConfig, RealTimeConfig, TrackingData};
use alvr_session::{
    ClientsideFoveationConfig, ClientsideFoveationMode, ClientsidePostProcessingConfig, CodecType,
    MediacodecProperty, PassthroughMode, UpscalingConfig,
};
use alvr_system_info::Platform;
use openxr as xr;
use std::{
    collections::VecDeque,
    ptr,
    rc::Rc,
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

// Time reserved for everything in a frame loop iteration other than waiting for a decoded frame
// (rendering, xrEndFrame(), event handling), until the loop has been measured.
const DEFAULT_RENDER_BUDGET: Duration = Duration::from_millis(6);
// The measured time excludes work the runtime does after xrEndFrame() returns.
const MIN_RENDER_BUDGET: Duration = Duration::from_millis(3);
// Slack between the end of an iteration and the next xrWaitFrame() wake-up.
const RENDER_BUDGET_MARGIN: Duration = Duration::from_millis(1);
// About one second of iterations. A high percentile follows the slower kind of iteration (a new
// frame costs more than a repeated one) while ignoring isolated hitches.
const RENDER_TIME_HISTORY_SIZE: usize = 128;
const MIN_RENDER_TIME_SAMPLES: usize = 32;
const RENDER_TIME_QUANTILE: f32 = 0.95;

// Measures how long a frame loop iteration takes apart from the wait for a decoded frame, so that
// the wait can use the rest of the display period and no more. An iteration runs from the return
// of xrWaitFrame() to the next call to it.
struct RenderBudget {
    samples: VecDeque<Duration>,
    sorted: Vec<Duration>,
    // Start of the current iteration and the time it spent waiting for a frame.
    iteration: Option<(Instant, Duration)>,
}

impl RenderBudget {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(RENDER_TIME_HISTORY_SIZE),
            sorted: Vec::with_capacity(RENDER_TIME_HISTORY_SIZE),
            iteration: None,
        }
    }

    fn report_wait(&mut self, iteration_start: Instant, waited: Duration) {
        self.iteration = Some((iteration_start, waited));
    }

    // Call right before xrWaitFrame(). Iterations that did not render are not sampled.
    fn report_iteration_end(&mut self, now: Instant) {
        if let Some((start, waited)) = self.iteration.take() {
            if self.samples.len() == RENDER_TIME_HISTORY_SIZE {
                self.samples.pop_front();
            }
            self.samples
                .push_back(now.saturating_duration_since(start).saturating_sub(waited));
        }
    }

    fn get(&mut self) -> Duration {
        if self.samples.len() < MIN_RENDER_TIME_SAMPLES {
            return DEFAULT_RENDER_BUDGET;
        }

        self.sorted.clear();
        self.sorted.extend(self.samples.iter().copied());
        let index =
            ((self.sorted.len() as f32 * RENDER_TIME_QUANTILE) as usize).min(self.sorted.len() - 1);
        let (_, render_time, _) = self.sorted.select_nth_unstable(index);

        (*render_time + RENDER_BUDGET_MARGIN).max(MIN_RENDER_BUDGET)
    }
}

// How long to wait for a decoded frame after xrBeginFrame().
//
// The whole iteration has to fit in one display period. xrWaitFrame() paces the loop to the
// display only as long as it is called before the next wake-up time: an iteration that waits a
// full period and then still has to render ends after it. The loop then follows the arrival of
// frames instead of the display, and the end of the wait, which is the deadline the streamer
// aligns its frames to, is no longer a fixed point of the display cycle.
//
// The predicted display time and current time share OpenXR's clock. A display closer than one
// period is a harder limit than the next wake-up, and the runtime needs time of its own before
// it: the wait keeps at least the default budget ahead of the display.
fn decoder_wait_timeout(
    frame_interval: Duration,
    vsync_time: Duration,
    xr_now: Option<Duration>,
    render_budget: Duration,
) -> Duration {
    let period_bound = frame_interval.saturating_sub(render_budget);

    if let Some(now) = xr_now {
        vsync_time
            .saturating_sub(now)
            .saturating_sub(render_budget.max(DEFAULT_RENDER_BUDGET))
            .min(period_bound)
    } else {
        period_bound
    }
}

pub struct ParsedStreamConfig {
    pub view_resolution: UVec2,
    pub refresh_rate_hint: f32,
    pub encoding_gamma: f32,
    pub enable_hdr: bool,
    pub passthrough: Option<PassthroughMode>,
    pub foveated_encoding_config: Option<AlvrFoveatedEncodingParams>,
    pub clientside_foveation_config: Option<ClientsideFoveationConfig>,
    pub clientside_post_processing: Option<ClientsidePostProcessingConfig>,
    pub upscaling: Option<UpscalingConfig>,
    pub force_software_decoder: bool,
    pub max_buffering_frames: f32,
    pub buffering_history_weight: f32,
    pub decoder_options: Vec<(String, MediacodecProperty)>,
    pub interaction_sources: InteractionSourcesConfig,
}

impl ParsedStreamConfig {
    pub fn new(config: &ClientStreamConfig) -> Self {
        Self {
            view_resolution: config.negotiated_config.view_resolution,
            refresh_rate_hint: config.negotiated_config.refresh_rate_hint,
            encoding_gamma: config.negotiated_config.encoding_gamma,
            enable_hdr: config.negotiated_config.enable_hdr,
            passthrough: config.settings.video.passthrough.as_option().cloned(),
            foveated_encoding_config: config.negotiated_config.foveated_encoding,
            clientside_foveation_config: config
                .settings
                .video
                .clientside_foveation
                .as_option()
                .cloned(),
            clientside_post_processing: config
                .settings
                .video
                .clientside_post_processing
                .as_option()
                .cloned(),
            upscaling: config.settings.video.upscaling.as_option().cloned(),
            force_software_decoder: config.settings.video.force_software_decoder,
            max_buffering_frames: config.settings.video.max_buffering_frames,
            buffering_history_weight: config.settings.video.buffering_history_weight,
            decoder_options: config.settings.video.mediacodec_extra_options.clone(),
            interaction_sources: InteractionSourcesConfig::new(config),
        }
    }
}

pub struct StreamContext {
    core_context: Arc<ClientCoreContext>,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_context: Arc<RwLock<InteractionContext>>,
    stage_reference_space: Arc<xr::Space>,
    view_reference_space: Arc<xr::Space>,
    swapchains: [xr::Swapchain<xr::OpenGlEs>; 2],
    last_good_video_frame_metadata: VideoFrameMetadata,
    input_thread: Option<JoinHandle<()>>,
    input_thread_running: Arc<RelaxedAtomic>,
    config: ParsedStreamConfig,
    target_view_resolution: UVec2,
    renderer: StreamRenderer,
    decoder: Option<(VideoDecoderConfig, VideoDecoderSource)>,
    use_custom_reprojection: bool,
    render_budget: RenderBudget,
}

impl StreamContext {
    pub fn new(
        core_ctx: Arc<ClientCoreContext>,
        xr_session: xr::Session<xr::OpenGlEs>,
        gfx_ctx: Rc<GraphicsContext>,
        interaction_ctx: Arc<RwLock<InteractionContext>>,
        config: ParsedStreamConfig,
    ) -> StreamContext {
        interaction_ctx
            .write()
            .select_sources(&config.interaction_sources);

        let xr_exts = xr_session.instance().exts();

        if xr_exts.fb_display_refresh_rate.is_some() {
            xr_session
                .request_display_refresh_rate(config.refresh_rate_hint)
                .unwrap();
        }

        let foveation_profile = if let Some(config) = &config.clientside_foveation_config
            && xr_exts.fb_swapchain_update_state.is_some()
            && xr_exts.fb_foveation.is_some()
            && xr_exts.fb_foveation_configuration.is_some()
        {
            let level;
            let dynamic;
            match config.mode {
                ClientsideFoveationMode::Static { level: lvl } => {
                    level = lvl;
                    dynamic = false;
                }
                ClientsideFoveationMode::Dynamic { max_level } => {
                    level = max_level;
                    dynamic = true;
                }
            };

            xr_session
                .create_foveation_profile(Some(xr::FoveationLevelProfile {
                    level: xr::FoveationLevelFB::from_raw(level as i32),
                    vertical_offset: config.vertical_offset_deg,
                    dynamic: xr::FoveationDynamicFB::from_raw(dynamic as i32),
                }))
                .ok()
        } else {
            None
        };

        let target_view_resolution = alvr_graphics::compute_target_view_resolution(
            config.view_resolution,
            &config.upscaling,
        );
        let format = graphics::swapchain_format(&gfx_ctx, &xr_session, config.enable_hdr);

        let swapchains = [
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
        ];

        let renderer = StreamRenderer::new(
            gfx_ctx,
            config.view_resolution,
            target_view_resolution,
            [
                swapchains[0]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
                swapchains[1]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
            ],
            format,
            config.foveated_encoding_config,
            !((core_ctx.platform().is_pico()
                || (core_ctx.platform() == Platform::SamsungGalaxyXR))
                && config.enable_hdr),
            // TODO: Find a driver heuristic for the limited range bug instead?
            core_ctx.platform() != Platform::SamsungGalaxyXR && !config.enable_hdr,
            config.encoding_gamma,
            config.upscaling.clone(),
        );

        {
            let int_ctx = interaction_ctx.read();
            core_ctx.send_active_interaction_profile(
                *HAND_LEFT_ID,
                int_ctx.hands_interaction[0].controllers_profile_id,
                int_ctx.hands_interaction[0].input_ids.clone(),
            );
            core_ctx.send_active_interaction_profile(
                *HAND_RIGHT_ID,
                int_ctx.hands_interaction[1].controllers_profile_id,
                int_ctx.hands_interaction[1].input_ids.clone(),
            );
        }

        let input_thread_running = Arc::new(RelaxedAtomic::new(false));

        let stage_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        let view_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        let mut this = StreamContext {
            use_custom_reprojection: core_ctx.platform().is_yvr(),
            core_context: core_ctx,
            xr_session,
            interaction_context: interaction_ctx,
            stage_reference_space,
            view_reference_space,
            swapchains,
            last_good_video_frame_metadata: VideoFrameMetadata {
                view_params: [ViewParams::DUMMY; 2],
                foveation_center_shifts: None,
            },
            input_thread: None,
            input_thread_running,
            config,
            target_view_resolution,
            renderer,
            decoder: None,
            render_budget: RenderBudget::new(),
        };

        this.update_reference_space();

        this
    }

    pub fn uses_passthrough(&self) -> bool {
        self.config.passthrough.is_some()
    }

    pub fn update_reference_space(&mut self) {
        self.input_thread_running.set(false);

        self.stage_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        self.view_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        self.core_context.send_playspace(
            self.xr_session
                .reference_space_bounds_rect(xr::ReferenceSpaceType::STAGE)
                .unwrap()
                .map(|a| Vec2::new(a.width, a.height)),
        );

        if let Some(running) = self.input_thread.take() {
            running.join().ok();
        }

        self.input_thread_running.set(true);

        self.input_thread = Some(thread::spawn({
            let core_ctx = Arc::clone(&self.core_context);
            let xr_session = self.xr_session.clone();
            let interaction_ctx = Arc::clone(&self.interaction_context);
            let stage_reference_space = Arc::clone(&self.stage_reference_space);
            let view_reference_space = Arc::clone(&self.view_reference_space);
            let refresh_rate = self.config.refresh_rate_hint;
            let running = Arc::clone(&self.input_thread_running);
            move || {
                stream_input_loop(
                    &core_ctx,
                    xr_session,
                    &interaction_ctx,
                    &stage_reference_space,
                    &view_reference_space,
                    refresh_rate,
                    running,
                )
            }
        }));
    }

    pub fn maybe_initialize_decoder(&mut self, codec: CodecType, config_nal: Vec<u8>) {
        let new_config = VideoDecoderConfig {
            codec,
            force_software_decoder: self.config.force_software_decoder,
            max_buffering_frames: self.config.max_buffering_frames,
            buffering_history_weight: self.config.buffering_history_weight,
            options: self.config.decoder_options.clone(),
            config_buffer: config_nal,
        };

        let maybe_config = if let Some((config, _)) = &self.decoder {
            (new_config != *config).then_some(new_config)
        } else {
            Some(new_config)
        };

        if let Some(config) = maybe_config {
            let (mut sink, source) = video_decoder::create_decoder(config.clone(), {
                let ctx = Arc::clone(&self.core_context);
                move |maybe_timestamp: Result<Duration>| match maybe_timestamp {
                    Ok(timestamp) => ctx.report_frame_decoded(timestamp),
                    Err(e) => ctx.report_fatal_decoder_error(&e.to_string()),
                }
            });
            self.decoder = Some((config, source));

            self.core_context.set_decoder_input_callback(Box::new(
                move |timestamp, buffer| -> bool { sink.push_nal(timestamp, buffer) },
            ));
        }
    }

    pub fn update_real_time_config(&mut self, config: &RealTimeConfig) {
        self.config.passthrough = config.passthrough.clone();
        self.config.clientside_post_processing = config.clientside_post_processing.clone();
    }

    /// Call right before xrWaitFrame(), so the time the frame loop needs besides waiting for a
    /// decoded frame can be measured.
    pub fn report_frame_loop_end(&mut self) {
        self.render_budget.report_iteration_end(Instant::now());
    }

    /// `frame_loop_start` is the time xrWaitFrame() returned for this frame.
    pub fn render(
        &mut self,
        frame_interval: Duration,
        vsync_time: Duration,
        frame_loop_start: Instant,
    ) -> (ProjectionLayerBuilder<'_>, Duration) {
        let xr_vsync_time = xr::Time::from_nanos(vsync_time.as_nanos() as _);

        // Sample the runtime clock immediately before waiting, preserving time for rendering and
        // xrEndFrame(). A zero timeout still dequeues an already available frame.
        let timeout = decoder_wait_timeout(
            frame_interval,
            vsync_time,
            crate::xr_runtime_now(self.xr_session.instance()).map(crate::from_xr_time),
            self.render_budget.get(),
        );
        let wait_start = Instant::now();
        let latch_deadline = wait_start + timeout;
        let (frame_result, dropped_frames) = match &mut self.decoder {
            Some((_, source)) => (
                source.get_frame_timeout(timeout),
                source.take_dropped_frames(),
            ),
            None => (None, 0),
        };
        self.render_budget
            .report_wait(frame_loop_start, wait_start.elapsed());
        // Reported for every refresh, including those without a new frame, so the streamer can
        // place its frames relative to this deadline.
        self.core_context.report_frame_latch(FrameLatch {
            deadline: latch_deadline,
            frame_timestamp: frame_result.map(|(timestamp, _)| timestamp),
            display_period: frame_interval,
            dropped_frames,
        });

        let (timestamp, frame_metadata, buffer_ptr) =
            if let Some((timestamp, buffer_ptr)) = frame_result {
                if let Some(metadata) = self.core_context.report_compositor_start(timestamp) {
                    self.last_good_video_frame_metadata = VideoFrameMetadata {
                        foveation_center_shifts: metadata
                            .foveation_center_shifts
                            .or(self.last_good_video_frame_metadata.foveation_center_shifts),
                        ..metadata
                    };
                }

                // Keep displaying new images even when their metadata is unavailable.
                (timestamp, self.last_good_video_frame_metadata, buffer_ptr)
            } else {
                (
                    vsync_time,
                    self.last_good_video_frame_metadata,
                    ptr::null_mut(),
                )
            };
        let view_params = frame_metadata.view_params;

        let left_swapchain_idx = self.swapchains[0].acquire_image().unwrap();
        let right_swapchain_idx = self.swapchains[1].acquire_image().unwrap();

        self.swapchains[0]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();
        self.swapchains[1]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();

        let (flags, maybe_views) = self
            .xr_session
            .locate_views(
                xr::ViewConfigurationType::PRIMARY_STEREO,
                xr_vsync_time,
                &self.stage_reference_space,
            )
            .unwrap();

        let current_headset_views = if flags.contains(xr::ViewStateFlags::ORIENTATION_VALID) {
            maybe_views
        } else {
            vec![crate::default_view(), crate::default_view()]
        };

        // The poses and FoVs we received from the PC runtime, which may differ and/or include
        // altered FoVs based on settings and view conversions done for canting.
        let input_view_params = view_params;
        let mut output_view_params = input_view_params;
        // Avoid passing invalid timestamp to runtime.
        // `timestamp` is generally a current vsync time, but may be repeated if frames are
        // dropped. Some runtimes dislike it if the timestamp is repeated for too long, so after
        // one second we begin presenting a lagged vsync time instead.
        let mut openxr_display_time =
            Duration::max(timestamp, vsync_time.saturating_sub(Duration::from_secs(1)));

        // (shinyquagsire23) I don't entirely trust runtimes to implement CompositionLayerProjectionView
        // correctly, but if we do trust them, avoid doing rotation ourselves. Otherwise, rerender.
        // Ex: YVR/PFDMR has issues with aspect ratio mismatches and passthrough compositing.
        if self.use_custom_reprojection {
            output_view_params = [
                ViewParams {
                    pose: crate::from_xr_pose(current_headset_views[0].pose),
                    fov: crate::from_xr_fov(current_headset_views[0].fov),
                },
                ViewParams {
                    pose: crate::from_xr_pose(current_headset_views[1].pose),
                    fov: crate::from_xr_fov(current_headset_views[1].fov),
                },
            ];

            openxr_display_time = vsync_time;
        }

        self.renderer.render(
            buffer_ptr,
            [
                StreamViewParams {
                    swapchain_index: left_swapchain_idx,
                    input_view_params: input_view_params[0],
                    output_view_params: output_view_params[0],
                },
                StreamViewParams {
                    swapchain_index: right_swapchain_idx,
                    input_view_params: input_view_params[1],
                    output_view_params: output_view_params[1],
                },
            ],
            self.config.passthrough.as_ref(),
            frame_metadata
                .foveation_center_shifts
                .map(|centers| centers.map(Vec2::from_array)),
        );

        self.swapchains[0].release_image().unwrap();
        self.swapchains[1].release_image().unwrap();

        if !buffer_ptr.is_null()
            && let Some(xr_now) = crate::xr_runtime_now(self.xr_session.instance())
        {
            self.core_context.report_submit(
                timestamp,
                vsync_time.saturating_sub(Duration::from_nanos(xr_now.as_nanos() as u64)),
            );
        }

        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: self.target_view_resolution.x as _,
                height: self.target_view_resolution.y as _,
            },
        };

        let clientside_post_processing = self
            .xr_session
            .instance()
            .exts()
            .fb_composition_layer_settings
            .and(self.config.clientside_post_processing.clone());

        let layer = ProjectionLayerBuilder::new(
            &self.stage_reference_space,
            [
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(output_view_params[0].pose))
                    .fov(crate::to_xr_fov(output_view_params[0].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[0])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(output_view_params[1].pose))
                    .fov(crate::to_xr_fov(output_view_params[1].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[1])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
            ],
            self.config
                .passthrough
                .clone()
                .map(|mode| ProjectionLayerAlphaConfig {
                    premultiplied: matches!(
                        mode,
                        PassthroughMode::Blend {
                            premultiplied_alpha: true,
                            ..
                        } | PassthroughMode::RgbChromaKey(_)
                            | PassthroughMode::HsvChromaKey(_)
                    ),
                }),
            clientside_post_processing,
        );

        (layer, openxr_display_time)
    }
}

impl Drop for StreamContext {
    fn drop(&mut self) {
        self.input_thread_running.set(false);
        self.input_thread.take().unwrap().join().ok();
    }
}

fn stream_input_loop(
    core_ctx: &ClientCoreContext,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_ctx: &RwLock<InteractionContext>,
    stage_reference_space: &xr::Space,
    view_reference_space: &xr::Space,
    refresh_rate: f32,
    running: Arc<RelaxedAtomic>,
) {
    // Tracking samples feed both the server's game poses and the frames' display timestamps.
    crate::declare_xr_thread(&xr_session, xr::AndroidThreadTypeKHR::APPLICATION_WORKER);

    let mut last_controller_poses = [Pose::IDENTITY; 2];
    let mut last_palm_poses = [Pose::IDENTITY; 2];
    let mut last_view_params = [ViewParams::DUMMY; 2];

    let mut deadline = Instant::now();
    let poll_interval = Duration::from_secs_f32(1.0 / (refresh_rate * 3.0));
    while running.value() {
        let int_ctx_lock = interaction_ctx.read();
        let int_ctx = &*int_ctx_lock;
        // Streaming related inputs are updated here. Make sure every input poll is done in this
        // thread
        if let Err(e) = xr_session.sync_actions(&[(&int_ctx.action_set).into()]) {
            error!("{e}");
            return;
        }

        let Some(now) = crate::xr_runtime_now(xr_session.instance()).map(crate::from_xr_time)
        else {
            error!("Cannot poll tracking: invalid time");
            return;
        };

        let target_time = now + core_ctx.get_total_prediction_offset();

        let Some((head_motion, local_views)) = interaction::get_head_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            view_reference_space,
            now,
            target_time,
            &last_view_params,
        ) else {
            continue;
        };

        if let Some(views) = local_views {
            core_ctx.send_view_params(views);
            last_view_params = views;
        }

        let mut device_motions = Vec::with_capacity(3);

        device_motions.push((*HEAD_ID, head_motion));

        let left_hand_data = crate::interaction::get_hand_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[0],
            &mut last_controller_poses[0],
            &mut last_palm_poses[0],
        );
        let right_hand_data = crate::interaction::get_hand_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[1],
            &mut last_controller_poses[1],
            &mut last_palm_poses[1],
        );

        // Note: When multimodal input is enabled, we are sure that when free hands are used
        // (not holding controllers) the controller data is None.
        if (int_ctx.multimodal_hands_enabled || left_hand_data.skeleton_joints.is_none())
            && let Some(motion) = left_hand_data.grip_motion
        {
            device_motions.push((*HAND_LEFT_ID, motion));
        }
        if (int_ctx.multimodal_hands_enabled || right_hand_data.skeleton_joints.is_none())
            && let Some(motion) = right_hand_data.grip_motion
        {
            device_motions.push((*HAND_RIGHT_ID, motion));
        }

        if int_ctx.multimodal_hands_enabled
            && let Some(detached_controller) = left_hand_data.detached_grip_motion
        {
            device_motions.push((*DETACHED_CONTROLLER_LEFT_ID, detached_controller));
        }
        if int_ctx.multimodal_hands_enabled
            && let Some(detached_controller) = right_hand_data.detached_grip_motion
        {
            device_motions.push((*DETACHED_CONTROLLER_RIGHT_ID, detached_controller));
        }

        let face = interaction::get_face_data(
            &xr_session,
            &int_ctx.face_sources,
            view_reference_space,
            now,
        );

        let body = int_ctx
            .body_source
            .as_ref()
            .and_then(|source| interaction::get_body_skeleton(source, stage_reference_space, now));

        if let Some(source) = &int_ctx.body_source {
            device_motions.append(&mut interaction::get_bd_motion_trackers(source, now));
        }

        drop(int_ctx_lock);

        // Even though the server is already adding the motion-to-photon latency, here we use
        // target_time as the poll_timestamp to compensate for the fact that video frames are sent
        // with the poll timestamp instead of the vsync time. This is to ensure correctness when
        // submitting frames to OpenXR. This won't cause any desync with the server because no time
        // sync step is performed between client and server.
        core_ctx.send_tracking(TrackingData {
            poll_timestamp: target_time,
            device_motions,
            hand_skeletons: [
                left_hand_data.skeleton_joints,
                right_hand_data.skeleton_joints,
            ],
            face,
            body,
        });

        let button_entries =
            interaction::update_buttons(&xr_session, &interaction_ctx.read().button_actions);
        if !button_entries.is_empty() {
            core_ctx.send_buttons(button_entries);
        }

        deadline += poll_interval;
        let now = Instant::now();
        if deadline > now {
            thread::sleep(deadline - now);
        } else {
            // Re-anchor after a stall so we don't send a back-to-back burst of tracking packets.
            deadline = now;
        }
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
    fn decoder_wait_leaves_the_render_budget_inside_the_display_period() {
        let now = Duration::from_secs(100);

        // The display is predicted more than two periods ahead, as on runtimes that reserve a
        // period for their own compositor. That does not leave more than a period to the loop.
        for (frame_interval, budget, expected) in [
            (us(8_333), ms(3), us(5_333)),
            (us(8_333), ms(6), us(2_333)),
            (us(11_111), ms(4), us(7_111)),
            (us(13_889), ms(6), us(7_889)),
        ] {
            assert_eq!(
                decoder_wait_timeout(frame_interval, now + ms(30), Some(now), budget),
                expected
            );
        }
    }

    #[test]
    fn decoder_wait_uses_the_remaining_display_time_when_it_is_shorter() {
        let frame_interval = us(8_333);
        let now = Duration::from_secs(100);

        // The same refresh rate can leave different amounts of time after xrWaitFrame(). The
        // default budget is kept before the display even when the loop needs less.
        assert_eq!(
            decoder_wait_timeout(frame_interval, now + ms(5), Some(now), ms(3)),
            Duration::ZERO
        );
        assert_eq!(
            decoder_wait_timeout(frame_interval, now + ms(8), Some(now), ms(3)),
            ms(2)
        );
        assert_eq!(
            decoder_wait_timeout(frame_interval, now + ms(12), Some(now), ms(3)),
            us(5_333)
        );
        // A loop that needs more than the default keeps that before the display.
        assert_eq!(
            decoder_wait_timeout(ms(20), now + ms(12), Some(now), ms(8)),
            ms(4)
        );
    }

    #[test]
    fn decoder_wait_stops_when_only_render_time_remains() {
        let frame_interval = us(8_333);
        let vsync_time = Duration::from_secs(100);

        for now in [
            vsync_time - ms(6),
            vsync_time - ms(4),
            vsync_time,
            vsync_time + ms(4),
        ] {
            assert_eq!(
                decoder_wait_timeout(frame_interval, vsync_time, Some(now), ms(6)),
                Duration::ZERO
            );
        }

        // A budget longer than the period leaves a single non-blocking dequeue attempt.
        assert_eq!(
            decoder_wait_timeout(frame_interval, vsync_time, Some(vsync_time - ms(30)), ms(9)),
            Duration::ZERO
        );
    }

    #[test]
    fn decoder_wait_uses_the_period_without_runtime_time() {
        let vsync_time = Duration::from_secs(100);

        assert_eq!(decoder_wait_timeout(ms(10), vsync_time, None, ms(6)), ms(4));
        assert_eq!(
            decoder_wait_timeout(ms(4), vsync_time, None, ms(6)),
            Duration::ZERO
        );
    }

    // One iteration that waited `waited` for a frame and needed `work` for everything else.
    fn iteration(budget: &mut RenderBudget, start: Instant, waited: Duration, work: Duration) {
        budget.report_wait(start, waited);
        budget.report_iteration_end(start + waited + work);
    }

    #[test]
    fn render_budget_starts_from_the_default() {
        let start = Instant::now();
        let mut budget = RenderBudget::new();
        assert_eq!(budget.get(), DEFAULT_RENDER_BUDGET);

        for _ in 0..MIN_RENDER_TIME_SAMPLES - 1 {
            iteration(&mut budget, start, ms(2), ms(4));
        }
        assert_eq!(budget.get(), DEFAULT_RENDER_BUDGET);

        iteration(&mut budget, start, ms(2), ms(4));
        assert_eq!(budget.get(), ms(4) + RENDER_BUDGET_MARGIN);
    }

    #[test]
    fn render_budget_excludes_the_wait_for_a_frame() {
        let start = Instant::now();
        let mut budget = RenderBudget::new();

        // However long the wait, the work after it takes the same time.
        for frame in 0..RENDER_TIME_HISTORY_SIZE as u64 {
            iteration(&mut budget, start, us(frame * 50), us(4_500));
        }

        assert_eq!(budget.get(), us(4_500) + RENDER_BUDGET_MARGIN);
    }

    #[test]
    fn render_budget_follows_the_slower_iterations_but_not_hitches() {
        let start = Instant::now();
        let mut budget = RenderBudget::new();

        // Three in four iterations repeat the previous frame, which is cheaper to render. A
        // decoder restart stalls a few iterations.
        for frame in 0..RENDER_TIME_HISTORY_SIZE {
            let work = match frame {
                40..43 => ms(150),
                _ if frame % 4 == 0 => us(5_200),
                _ => us(3_400),
            };
            iteration(&mut budget, start, Duration::ZERO, work);
        }

        assert_eq!(budget.get(), us(5_200) + RENDER_BUDGET_MARGIN);
    }

    #[test]
    fn render_budget_has_a_floor_and_forgets_old_iterations() {
        let start = Instant::now();
        let mut budget = RenderBudget::new();

        for _ in 0..RENDER_TIME_HISTORY_SIZE {
            iteration(&mut budget, start, Duration::ZERO, ms(7));
        }
        assert_eq!(budget.get(), ms(7) + RENDER_BUDGET_MARGIN);

        for _ in 0..RENDER_TIME_HISTORY_SIZE {
            iteration(&mut budget, start, Duration::ZERO, us(500));
        }
        assert_eq!(budget.get(), MIN_RENDER_BUDGET);
    }

    #[test]
    fn iterations_that_did_not_render_are_not_sampled() {
        let start = Instant::now();
        let mut budget = RenderBudget::new();

        for _ in 0..MIN_RENDER_TIME_SAMPLES {
            iteration(&mut budget, start, Duration::ZERO, ms(4));
        }
        // The runtime asked not to render for a while: only xrWaitFrame() calls in between.
        budget.report_iteration_end(start + Duration::from_secs(5));
        budget.report_iteration_end(start + Duration::from_secs(6));

        assert_eq!(budget.samples.len(), MIN_RENDER_TIME_SAMPLES);
        assert_eq!(budget.get(), ms(4) + RENDER_BUDGET_MARGIN);
    }
}
