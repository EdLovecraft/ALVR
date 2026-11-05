//! Lets the platform layer raise the scheduling priority of client threads on the video path, for
//! example with XR_KHR_android_thread_settings. Preemption of these threads shows up directly as
//! frames missing the compositor latch.

use alvr_common::parking_lot::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CriticalThreadKind {
    /// Receives stream packets and feeds the video decoder.
    NetworkReceive,
    /// Moves decoded images from the video decoder to the compositor queue.
    Decoder,
}

pub type CriticalThreadHook = dyn Fn(CriticalThreadKind) + Send + Sync;

static HOOK: RwLock<Option<Box<CriticalThreadHook>>> = RwLock::new(None);

/// Installs (or clears, with `None`) the callback invoked on each critical thread when it starts.
/// Threads that started before the hook was installed are not reported again.
///
/// The hook runs while a read lock is held, so clearing it waits for in-flight calls. A hook may
/// therefore capture a handle that is destroyed right after the hook is cleared.
pub fn set_critical_thread_hook(hook: Option<Box<CriticalThreadHook>>) {
    *HOOK.write() = hook;
}

/// Reports the calling thread to the installed hook, if any.
pub fn declare_current_thread(kind: CriticalThreadKind) {
    if let Some(hook) = &*HOOK.read() {
        hook(kind);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn declarations_reach_the_installed_hook_only() {
        let declared = Arc::new(Mutex::new(Vec::new()));

        declare_current_thread(CriticalThreadKind::Decoder);

        set_critical_thread_hook(Some(Box::new({
            let declared = Arc::clone(&declared);
            move |kind| declared.lock().unwrap().push(kind)
        })));
        declare_current_thread(CriticalThreadKind::NetworkReceive);
        declare_current_thread(CriticalThreadKind::Decoder);

        set_critical_thread_hook(None);
        declare_current_thread(CriticalThreadKind::Decoder);

        assert_eq!(
            *declared.lock().unwrap(),
            [
                CriticalThreadKind::NetworkReceive,
                CriticalThreadKind::Decoder
            ]
        );
    }
}
