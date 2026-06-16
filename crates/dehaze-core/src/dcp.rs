use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct RgbImageF32 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<[f32; 3]>,
}

impl RgbImageF32 {
    pub fn new(width: usize, height: usize, data: Vec<[f32; 3]>) -> Self {
        assert_eq!(width * height, data.len());
        Self {
            width,
            height,
            data,
        }
    }

    pub fn blank(width: usize, height: usize, value: [f32; 3]) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 3] {
        self.data[self.idx(x, y)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: [f32; 3]) {
        let idx = self.idx(x, y);
        self.data[idx] = value;
    }

    pub fn luma(&self) -> Vec<f32> {
        self.luma_with_mode(SimdMode::Auto)
    }

    pub fn luma_with_mode(&self, mode: SimdMode) -> Vec<f32> {
        if mode.allow_avx2() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                return luma_avx2(&self.data);
            }
        }
        luma_scalar(&self.data)
    }
}

fn luma_scalar(data: &[[f32; 3]]) -> Vec<f32> {
    data.iter()
        .map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn luma_avx2(data: &[[f32; 3]]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = data.len();
    let mut out = vec![0.0; len];
    let base = data.as_ptr() as *const f32;
    let mut i = 0usize;
    let wr = _mm256_set1_ps(0.299);
    let wg = _mm256_set1_ps(0.587);
    let wb = _mm256_set1_ps(0.114);
    while i + 8 <= len {
        let start = (i * 3) as i32;
        let r_idx = _mm256_setr_epi32(
            start,
            start + 3,
            start + 6,
            start + 9,
            start + 12,
            start + 15,
            start + 18,
            start + 21,
        );
        let g_idx = _mm256_add_epi32(r_idx, _mm256_set1_epi32(1));
        let b_idx = _mm256_add_epi32(r_idx, _mm256_set1_epi32(2));
        let r = _mm256_i32gather_ps(base, r_idx, 4);
        let g = _mm256_i32gather_ps(base, g_idx, 4);
        let b = _mm256_i32gather_ps(base, b_idx, 4);
        let lum = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r, wr), _mm256_mul_ps(g, wg)),
            _mm256_mul_ps(b, wb),
        );
        _mm256_storeu_ps(out.as_mut_ptr().add(i), lum);
        i += 8;
    }
    while i < len {
        let p = data[i];
        out[i] = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        i += 1;
    }
    out
}

#[derive(Clone, Debug)]
pub struct DehazeParams {
    pub patch_size: usize,
    pub omega: f32,
    pub t_min: f32,
    pub refine_radius: usize,
    pub guided_eps: f32,
    pub aggressiveness: f32,
    pub sky_protect: bool,
    pub highlight_protect: bool,
    pub simd: SimdMode,
}

impl Default for DehazeParams {
    fn default() -> Self {
        Self {
            patch_size: 15,
            omega: 0.96,
            t_min: 0.10,
            refine_radius: 12,
            guided_eps: 1e-3,
            aggressiveness: 0.20,
            sky_protect: true,
            highlight_protect: true,
            simd: SimdMode::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdMode {
    Auto,
    Scalar,
    Avx2,
}

impl SimdMode {
    fn allow_avx2(self) -> bool {
        match self {
            SimdMode::Auto => cpu_supports_avx2(),
            SimdMode::Scalar => false,
            SimdMode::Avx2 => cpu_supports_avx2(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DehazeOutput {
    pub image: RgbImageF32,
    pub dark_channel: Vec<f32>,
    pub transmission: Vec<f32>,
    pub airlight: [f32; 3],
    pub protection_mask: Vec<f32>,
    pub mean_transmission: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MotionParams {
    pub search_radius: usize,
    pub block_size: usize,
    pub pyramid_levels: usize,
}

impl Default for MotionParams {
    fn default() -> Self {
        Self {
            search_radius: 6,
            block_size: 12,
            pyramid_levels: 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TemporalParams {
    pub temporal_weight: f32,
    pub occlusion_threshold: f32,
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            temporal_weight: 0.65,
            occlusion_threshold: 0.18,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TemporalDiagnostics {
    pub frame_index: usize,
    pub airlight: [f32; 3],
    pub raw_airlight: [f32; 3],
    pub mean_transmission: f32,
    pub flicker: f32,
    pub mean_motion_error: f32,
    pub scene_reset: bool,
    pub debug: Option<TemporalDebug>,
}

#[derive(Clone, Debug)]
pub struct TemporalDebug {
    pub width: usize,
    pub height: usize,
    pub transmission: Vec<f32>,
    pub warped_transmission: Vec<f32>,
    pub protection_mask: Vec<f32>,
    pub motion_magnitude: Vec<f32>,
    pub motion_error: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct AirlightSmoother {
    beta: f32,
    current: Option<[f32; 3]>,
    reset_threshold: f32,
    previous_stats: Option<FrameStats>,
}

impl AirlightSmoother {
    pub fn new(beta: f32, reset_threshold: f32) -> Self {
        Self {
            beta: beta.clamp(0.0, 1.0),
            current: None,
            reset_threshold,
            previous_stats: None,
        }
    }

    pub fn update(&mut self, next: [f32; 3]) -> [f32; 3] {
        let Some(prev) = self.current else {
            self.current = Some(next);
            return next;
        };

        let delta =
            ((next[0] - prev[0]).abs() + (next[1] - prev[1]).abs() + (next[2] - prev[2]).abs())
                / 3.0;
        let smoothed = if delta > self.reset_threshold {
            next
        } else {
            [
                self.beta * next[0] + (1.0 - self.beta) * prev[0],
                self.beta * next[1] + (1.0 - self.beta) * prev[1],
                self.beta * next[2] + (1.0 - self.beta) * prev[2],
            ]
        };
        self.current = Some(smoothed);
        smoothed
    }

    pub fn update_with_stats(&mut self, next: [f32; 3], stats: FrameStats) -> ([f32; 3], bool) {
        let reset = self
            .previous_stats
            .map(|prev| prev.distance(stats) > self.reset_threshold)
            .unwrap_or(false);
        self.previous_stats = Some(stats);

        if reset {
            self.current = Some(next);
            (next, true)
        } else {
            (self.update(next), false)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FrameStats {
    mean_luma: f32,
    contrast: f32,
    bright_fraction: f32,
}

impl FrameStats {
    pub fn from_image(input: &RgbImageF32) -> Self {
        let luma = input.luma();
        let n = luma.len().max(1) as f32;
        let mean_luma = luma.iter().sum::<f32>() / n;
        let contrast = (luma.iter().map(|v| (v - mean_luma).powi(2)).sum::<f32>() / n).sqrt();
        let bright_fraction = luma.iter().filter(|&&v| v > 0.78).count() as f32 / n;
        Self {
            mean_luma,
            contrast,
            bright_fraction,
        }
    }

    fn distance(self, other: Self) -> f32 {
        (self.mean_luma - other.mean_luma).abs()
            + 0.8 * (self.contrast - other.contrast).abs()
            + 0.5 * (self.bright_fraction - other.bright_fraction).abs()
    }
}

pub fn original_dcp(input: &RgbImageF32, params: &DehazeParams) -> DehazeOutput {
    let mut baseline = params.clone();
    baseline.aggressiveness = 0.0;
    baseline.sky_protect = false;
    baseline.highlight_protect = false;
    let dark_channel = dark_channel(input, baseline.patch_size);
    let airlight = estimate_airlight(input, &dark_channel);
    let rough_t = estimate_transmission(input, airlight, baseline.patch_size, baseline.omega);
    let refined = guided_filter_rgb(input, &rough_t, baseline.refine_radius, baseline.guided_eps);
    let transmission = refined
        .into_iter()
        .map(|t| t.clamp(baseline.t_min, 1.0))
        .collect::<Vec<_>>();
    let image = recover(input, airlight, &transmission, baseline.t_min);
    DehazeOutput {
        image,
        dark_channel,
        transmission: transmission.clone(),
        airlight,
        protection_mask: vec![0.0; input.width * input.height],
        mean_transmission: mean(&transmission),
    }
}

pub fn improved_dcp(input: &RgbImageF32, params: &DehazeParams) -> DehazeOutput {
    dehaze_frame_cap_fusion(input, params)
}

pub fn dehaze_frame_cap_fusion(input: &RgbImageF32, params: &DehazeParams) -> DehazeOutput {
    let dark_channel = dark_channel(input, params.patch_size);
    let protection_mask = sky_highlight_mask(input, params.sky_protect, params.highlight_protect);
    let airlight = estimate_airlight_robust(input, &dark_channel, &protection_mask);
    dehaze_frame_with_airlight_cap_fusion(input, params, airlight, dark_channel)
}

pub fn dehaze_frame_with_airlight_cap_fusion(
    input: &RgbImageF32,
    params: &DehazeParams,
    airlight: [f32; 3],
    dark_channel: Vec<f32>,
) -> DehazeOutput {
    let protection_mask = sky_highlight_mask(input, params.sky_protect, params.highlight_protect);
    dehaze_frame_with_airlight_cap_fusion_and_mask(
        input,
        params,
        airlight,
        dark_channel,
        protection_mask,
    )
}

pub fn dehaze_frame_with_airlight_cap_fusion_and_mask(
    input: &RgbImageF32,
    params: &DehazeParams,
    airlight: [f32; 3],
    dark_channel: Vec<f32>,
    protection_mask: Vec<f32>,
) -> DehazeOutput {
    let adaptive = adaptive_params(input, params);
    let near_t = estimate_transmission_with_mode(
        input,
        airlight,
        params.patch_size,
        adaptive.omega,
        params.simd,
    );
    let cap_t = color_attenuation_transmission_prior(input);
    let edge = edge_strength(input);

    let fused = near_t
        .iter()
        .zip(cap_t.iter())
        .zip(edge.iter())
        .zip(protection_mask.iter())
        .map(|(((&near, &cap), &edge), &mask)| {
            let edge_weight = edge.clamp(0.0, 1.0);
            let cap_agreement = (1.0 - (cap - near).abs() * 4.0).clamp(0.0, 1.0);
            let cap_weight =
                0.16 * cap_agreement * (1.0 - 0.80 * mask) * (1.0 - 0.35 * edge_weight);
            let near_weight = (1.0 - cap_weight).max(0.0);
            (near * near_weight + cap * cap_weight).clamp(0.0, 1.0)
        })
        .collect::<Vec<_>>();

    let refined = guided_filter_rgb(input, &fused, params.refine_radius, params.guided_eps);
    let transmission = refined
        .into_iter()
        .zip(protection_mask.iter())
        .map(|(t, &mask)| {
            let mask = effective_protection(mask, params.aggressiveness);
            let protected_min = adaptive.t_min.max(params.t_min + 0.07 * mask);
            t.clamp(protected_min, 1.0)
        })
        .collect::<Vec<_>>();
    let image = recover_protected(
        input,
        airlight,
        &transmission,
        &protection_mask,
        adaptive.t_min,
        params.aggressiveness,
    );
    let mean_transmission = mean(&transmission);

    DehazeOutput {
        image,
        dark_channel,
        transmission,
        airlight,
        protection_mask,
        mean_transmission,
    }
}

pub fn dark_channel(input: &RgbImageF32, patch_size: usize) -> Vec<f32> {
    let radius = patch_size.max(1) / 2;
    let min_rgb = input
        .data
        .iter()
        .map(|p| p[0].min(p[1]).min(p[2]))
        .collect::<Vec<_>>();
    min_filter_gray(input.width, input.height, &min_rgb, radius)
}

pub fn estimate_airlight(input: &RgbImageF32, dark: &[f32]) -> [f32; 3] {
    let empty_mask = vec![0.0; dark.len()];
    estimate_airlight_robust(input, dark, &empty_mask)
}

pub fn estimate_airlight_robust(
    input: &RgbImageF32,
    dark: &[f32],
    protection_mask: &[f32],
) -> [f32; 3] {
    let mut indices = (0..dark.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| {
        dark[b]
            .partial_cmp(&dark[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sample_count = ((dark.len() as f32) * 0.01).ceil().max(8.0) as usize;
    let mut best_idx = indices[0];
    let mut best_score = f32::NEG_INFINITY;

    for &idx in indices.iter().take(sample_count.min(indices.len())) {
        let p = input.data[idx];
        let max_c = p[0].max(p[1]).max(p[2]);
        let min_c = p[0].min(p[1]).min(p[2]);
        let brightness = (p[0] + p[1] + p[2]) / 3.0;
        let saturation = if max_c > 1e-6 {
            (max_c - min_c) / max_c
        } else {
            0.0
        };
        let white_clip_penalty = if max_c > 0.97 && saturation < 0.08 {
            0.35
        } else {
            0.0
        };
        let protected = protection_mask.get(idx).copied().unwrap_or(0.0);
        let score = 0.50 * dark[idx] + 0.42 * brightness
            - 0.24 * saturation
            - 0.50 * protected
            - white_clip_penalty;
        if score > best_score {
            best_idx = idx;
            best_score = score;
        }
    }

    let best = input.data[best_idx];
    [
        best[0].max(1.0 / 255.0),
        best[1].max(1.0 / 255.0),
        best[2].max(1.0 / 255.0),
    ]
}

pub fn estimate_transmission(
    input: &RgbImageF32,
    airlight: [f32; 3],
    patch_size: usize,
    omega: f32,
) -> Vec<f32> {
    estimate_transmission_with_mode(input, airlight, patch_size, omega, SimdMode::Auto)
}

fn estimate_transmission_with_mode(
    input: &RgbImageF32,
    airlight: [f32; 3],
    patch_size: usize,
    omega: f32,
    mode: SimdMode,
) -> Vec<f32> {
    let radius = patch_size.max(1) / 2;
    let min_normalized = normalized_min_channel(input, airlight);
    let dark = min_filter_gray(input.width, input.height, &min_normalized, radius);
    transmission_from_dark_with_mode(&dark, omega, mode)
}

fn normalized_min_channel(input: &RgbImageF32, airlight: [f32; 3]) -> Vec<f32> {
    let inv_airlight = [
        1.0 / airlight[0].max(1.0 / 255.0),
        1.0 / airlight[1].max(1.0 / 255.0),
        1.0 / airlight[2].max(1.0 / 255.0),
    ];
    input
        .data
        .iter()
        .map(|p| {
            (p[0] * inv_airlight[0])
                .min(p[1] * inv_airlight[1])
                .min(p[2] * inv_airlight[2])
        })
        .collect()
}

fn transmission_from_dark_with_mode(dark: &[f32], omega: f32, mode: SimdMode) -> Vec<f32> {
    if mode.allow_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return transmission_from_dark_avx2(dark, omega);
        }
    }
    transmission_from_dark_scalar(dark, omega)
}

fn transmission_from_dark_scalar(dark: &[f32], omega: f32) -> Vec<f32> {
    dark.iter()
        .map(|&v| (1.0 - omega * v).clamp(0.0, 1.0))
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transmission_from_dark_avx2(dark: &[f32], omega: f32) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = dark.len();
    let mut out = vec![0.0; len];
    let mut i = 0usize;
    let ones = _mm256_set1_ps(1.0);
    let zeros = _mm256_setzero_ps();
    let omega_v = _mm256_set1_ps(omega);
    while i + 8 <= len {
        let d = _mm256_loadu_ps(dark.as_ptr().add(i));
        let t = _mm256_sub_ps(ones, _mm256_mul_ps(omega_v, d));
        let t = _mm256_min_ps(ones, _mm256_max_ps(zeros, t));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), t);
        i += 8;
    }
    while i < len {
        out[i] = (1.0 - omega * dark[i]).clamp(0.0, 1.0);
        i += 1;
    }
    out
}

pub fn recover(
    input: &RgbImageF32,
    airlight: [f32; 3],
    transmission: &[f32],
    t_min: f32,
) -> RgbImageF32 {
    let data = input
        .data
        .iter()
        .zip(transmission.iter())
        .map(|(p, &t)| {
            let tt = t.max(t_min);
            [
                ((p[0] - airlight[0]) / tt + airlight[0]).clamp(0.0, 1.0),
                ((p[1] - airlight[1]) / tt + airlight[1]).clamp(0.0, 1.0),
                ((p[2] - airlight[2]) / tt + airlight[2]).clamp(0.0, 1.0),
            ]
        })
        .collect();
    RgbImageF32::new(input.width, input.height, data)
}

fn recover_protected(
    input: &RgbImageF32,
    airlight: [f32; 3],
    transmission: &[f32],
    protection_mask: &[f32],
    t_min: f32,
    aggressiveness: f32,
) -> RgbImageF32 {
    let aggressiveness = aggressiveness.clamp(0.0, 1.0);
    let data = input
        .data
        .iter()
        .zip(transmission.iter())
        .zip(protection_mask.iter())
        .map(|((p, &t), &mask)| {
            let tt = t.max(t_min);
            let mask = effective_protection(mask, aggressiveness);
            let gain = 1.0 - 0.30 * mask;
            [
                (airlight[0] + gain * (p[0] - airlight[0]) / tt).clamp(0.0, 1.0),
                (airlight[1] + gain * (p[1] - airlight[1]) / tt).clamp(0.0, 1.0),
                (airlight[2] + gain * (p[2] - airlight[2]) / tt).clamp(0.0, 1.0),
            ]
        })
        .collect();
    RgbImageF32::new(input.width, input.height, data)
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveParams {
    omega: f32,
    t_min: f32,
}

fn adaptive_params(input: &RgbImageF32, params: &DehazeParams) -> AdaptiveParams {
    let stats = FrameStats::from_image(input);
    let haze_score =
        (stats.mean_luma * 0.55 + (0.22 - stats.contrast).max(0.0) * 1.4).clamp(0.0, 1.0);
    let aggressiveness = params.aggressiveness.clamp(0.0, 1.0);
    let omega_floor = 0.88 + 0.04 * aggressiveness;
    let t_guard = 0.05 * (1.0 - aggressiveness);
    AdaptiveParams {
        omega: (params.omega * (omega_floor + 0.08 * haze_score)).clamp(0.84, 1.0),
        t_min: (params.t_min + t_guard * (1.0 - haze_score)).clamp(params.t_min, 0.20),
    }
}

#[inline]
fn effective_protection(mask: f32, aggressiveness: f32) -> f32 {
    mask.clamp(0.0, 1.0) * (1.0 - 0.45 * aggressiveness.clamp(0.0, 1.0))
}

pub fn sky_highlight_mask(
    input: &RgbImageF32,
    sky_protect: bool,
    highlight_protect: bool,
) -> Vec<f32> {
    let luma = input.luma();
    let variance = local_variance(input.width, input.height, &luma, 4);
    input
        .data
        .iter()
        .zip(variance.iter())
        .map(|(p, &var)| {
            let max_c = p[0].max(p[1]).max(p[2]);
            let min_c = p[0].min(p[1]).min(p[2]);
            let brightness = (p[0] + p[1] + p[2]) / 3.0;
            let saturation = if max_c > 1e-6 {
                (max_c - min_c) / max_c
            } else {
                0.0
            };
            let blue_dominant = p[2] > p[0] * 1.04 && p[2] > p[1] * 1.02;
            let sky = sky_protect
                && brightness > 0.56
                && saturation < 0.28
                && var < 0.012
                && (blue_dominant || p[2] > 0.45);
            let highlight = highlight_protect && brightness > 0.90 && saturation < 0.14;
            if highlight {
                1.0
            } else if sky {
                0.65
            } else {
                0.0
            }
        })
        .collect()
}

pub fn synthesize_haze(input: &RgbImageF32, beta: f32, airlight: [f32; 3]) -> RgbImageF32 {
    let denom = (input.width + input.height).max(1) as f32;
    let mut out = RgbImageF32::blank(input.width, input.height, [0.0; 3]);

    for y in 0..input.height {
        for x in 0..input.width {
            let p = input.get(x, y);
            let depth = ((x + y) as f32 / denom).clamp(0.0, 1.0);
            let t = (-beta * depth).exp().clamp(0.05, 1.0);
            out.set(
                x,
                y,
                [
                    p[0] * t + airlight[0] * (1.0 - t),
                    p[1] * t + airlight[1] * (1.0 - t),
                    p[2] * t + airlight[2] * (1.0 - t),
                ],
            );
        }
    }

    out
}

pub fn box_filter_gray(width: usize, height: usize, input: &[f32], radius: usize) -> Vec<f32> {
    assert_eq!(width * height, input.len());
    if radius == 0 {
        return input.to_vec();
    }

    let integral = integral_image(width, height, input);
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let y0 = y.saturating_sub(radius);
            let x1 = (x + radius).min(width - 1);
            let y1 = (y + radius).min(height - 1);
            let sum = integral_sum(&integral, width, x0, y0, x1, y1);
            let count = (x1 - x0 + 1) * (y1 - y0 + 1);
            out[y * width + x] = sum / count as f32;
        }
    }
    out
}

pub fn min_filter_gray(width: usize, height: usize, input: &[f32], radius: usize) -> Vec<f32> {
    assert_eq!(width * height, input.len());
    if radius == 0 {
        return input.to_vec();
    }
    let horizontal = sliding_min_horizontal(width, height, input, radius);
    sliding_min_vertical(width, height, &horizontal, radius)
}

fn sliding_min_horizontal(width: usize, height: usize, input: &[f32], radius: usize) -> Vec<f32> {
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        let row = &input[y * width..(y + 1) * width];
        let mut deque: VecDeque<usize> = VecDeque::new();
        let mut right_added = 0usize;
        for x in 0..width {
            let right = (x + radius).min(width - 1);
            while right_added <= right {
                while let Some(&back) = deque.back() {
                    if row[back] <= row[right_added] {
                        break;
                    }
                    deque.pop_back();
                }
                deque.push_back(right_added);
                right_added += 1;
            }
            let left = x.saturating_sub(radius);
            while let Some(&front) = deque.front() {
                if front >= left {
                    break;
                }
                deque.pop_front();
            }
            out[y * width + x] = row[*deque.front().expect("non-empty min window")];
        }
    }
    out
}

fn sliding_min_vertical(width: usize, height: usize, input: &[f32], radius: usize) -> Vec<f32> {
    let mut out = vec![0.0; width * height];
    for x in 0..width {
        let mut deque: VecDeque<usize> = VecDeque::new();
        let mut bottom_added = 0usize;
        for y in 0..height {
            let bottom = (y + radius).min(height - 1);
            while bottom_added <= bottom {
                while let Some(&back) = deque.back() {
                    if input[back * width + x] <= input[bottom_added * width + x] {
                        break;
                    }
                    deque.pop_back();
                }
                deque.push_back(bottom_added);
                bottom_added += 1;
            }
            let top = y.saturating_sub(radius);
            while let Some(&front) = deque.front() {
                if front >= top {
                    break;
                }
                deque.pop_front();
            }
            out[y * width + x] = input[*deque.front().expect("non-empty min window") * width + x];
        }
    }
    out
}

fn integral_image(width: usize, height: usize, input: &[f32]) -> Vec<f32> {
    let stride = width + 1;
    let mut integral = vec![0.0; (width + 1) * (height + 1)];
    for y in 0..height {
        let mut row_sum = 0.0;
        for x in 0..width {
            row_sum += input[y * width + x];
            let dst = (y + 1) * stride + x + 1;
            integral[dst] = integral[y * stride + x + 1] + row_sum;
        }
    }
    integral
}

#[inline]
fn integral_sum(integral: &[f32], width: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
    let stride = width + 1;
    let ax = x0;
    let ay = y0;
    let bx = x1 + 1;
    let by = y1 + 1;
    integral[by * stride + bx] - integral[ay * stride + bx] - integral[by * stride + ax]
        + integral[ay * stride + ax]
}

pub fn guided_filter_rgb(guide: &RgbImageF32, input: &[f32], radius: usize, eps: f32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let n = guide.width * guide.height;
    assert_eq!(input.len(), n);

    let ir = guide.data.iter().map(|p| p[0]).collect::<Vec<_>>();
    let ig = guide.data.iter().map(|p| p[1]).collect::<Vec<_>>();
    let ib = guide.data.iter().map(|p| p[2]).collect::<Vec<_>>();
    let mean_r = box_filter_gray(guide.width, guide.height, &ir, radius);
    let mean_g = box_filter_gray(guide.width, guide.height, &ig, radius);
    let mean_b = box_filter_gray(guide.width, guide.height, &ib, radius);
    let mean_p = box_filter_gray(guide.width, guide.height, input, radius);

    let rr = ir.iter().map(|v| v * v).collect::<Vec<_>>();
    let gg = ig.iter().map(|v| v * v).collect::<Vec<_>>();
    let bb = ib.iter().map(|v| v * v).collect::<Vec<_>>();
    let rg = ir
        .iter()
        .zip(ig.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();
    let rb = ir
        .iter()
        .zip(ib.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();
    let gb = ig
        .iter()
        .zip(ib.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();
    let rp = ir
        .iter()
        .zip(input.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();
    let gp = ig
        .iter()
        .zip(input.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();
    let bp = ib
        .iter()
        .zip(input.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();

    let mean_rr = box_filter_gray(guide.width, guide.height, &rr, radius);
    let mean_gg = box_filter_gray(guide.width, guide.height, &gg, radius);
    let mean_bb = box_filter_gray(guide.width, guide.height, &bb, radius);
    let mean_rg = box_filter_gray(guide.width, guide.height, &rg, radius);
    let mean_rb = box_filter_gray(guide.width, guide.height, &rb, radius);
    let mean_gb = box_filter_gray(guide.width, guide.height, &gb, radius);
    let mean_rp = box_filter_gray(guide.width, guide.height, &rp, radius);
    let mean_gp = box_filter_gray(guide.width, guide.height, &gp, radius);
    let mean_bp = box_filter_gray(guide.width, guide.height, &bp, radius);

    let mut ar = vec![0.0; n];
    let mut ag = vec![0.0; n];
    let mut ab = vec![0.0; n];
    let mut b = vec![0.0; n];
    for i in 0..n {
        let cov = [
            [
                mean_rr[i] - mean_r[i] * mean_r[i] + eps,
                mean_rg[i] - mean_r[i] * mean_g[i],
                mean_rb[i] - mean_r[i] * mean_b[i],
            ],
            [
                mean_rg[i] - mean_r[i] * mean_g[i],
                mean_gg[i] - mean_g[i] * mean_g[i] + eps,
                mean_gb[i] - mean_g[i] * mean_b[i],
            ],
            [
                mean_rb[i] - mean_r[i] * mean_b[i],
                mean_gb[i] - mean_g[i] * mean_b[i],
                mean_bb[i] - mean_b[i] * mean_b[i] + eps,
            ],
        ];
        let cov_ip = [
            mean_rp[i] - mean_r[i] * mean_p[i],
            mean_gp[i] - mean_g[i] * mean_p[i],
            mean_bp[i] - mean_b[i] * mean_p[i],
        ];
        let a = solve_3x3(cov, cov_ip).unwrap_or([0.0; 3]);
        ar[i] = a[0];
        ag[i] = a[1];
        ab[i] = a[2];
        b[i] = mean_p[i] - ar[i] * mean_r[i] - ag[i] * mean_g[i] - ab[i] * mean_b[i];
    }

    let mean_ar = box_filter_gray(guide.width, guide.height, &ar, radius);
    let mean_ag = box_filter_gray(guide.width, guide.height, &ag, radius);
    let mean_ab = box_filter_gray(guide.width, guide.height, &ab, radius);
    let mean_b2 = box_filter_gray(guide.width, guide.height, &b, radius);
    (0..n)
        .map(|i| mean_ar[i] * ir[i] + mean_ag[i] * ig[i] + mean_ab[i] * ib[i] + mean_b2[i])
        .collect()
}

fn solve_3x3(mut a: [[f32; 3]; 3], mut b: [f32; 3]) -> Option<[f32; 3]> {
    for i in 0..3 {
        let mut pivot = i;
        for row in (i + 1)..3 {
            if a[row][i].abs() > a[pivot][i].abs() {
                pivot = row;
            }
        }
        if a[pivot][i].abs() < 1e-8 {
            return None;
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        let div = a[i][i];
        for col in i..3 {
            a[i][col] /= div;
        }
        b[i] /= div;
        for row in 0..3 {
            if row == i {
                continue;
            }
            let factor = a[row][i];
            for col in i..3 {
                a[row][col] -= factor * a[i][col];
            }
            b[row] -= factor * b[i];
        }
    }
    Some(b)
}

fn local_variance(width: usize, height: usize, input: &[f32], radius: usize) -> Vec<f32> {
    let mean_v = box_filter_gray(width, height, input, radius);
    let sq = input.iter().map(|v| v * v).collect::<Vec<_>>();
    let mean_sq = box_filter_gray(width, height, &sq, radius);
    mean_sq
        .into_iter()
        .zip(mean_v)
        .map(|(sq, m)| (sq - m * m).max(0.0))
        .collect()
}

fn edge_strength(input: &RgbImageF32) -> Vec<f32> {
    let luma = input.luma();
    let mut out = vec![0.0; input.width * input.height];
    for y in 0..input.height {
        for x in 0..input.width {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(input.width - 1);
            let yu = y.saturating_sub(1);
            let yd = (y + 1).min(input.height - 1);
            let gx = (luma[y * input.width + xr] - luma[y * input.width + xl]).abs();
            let gy = (luma[yd * input.width + x] - luma[yu * input.width + x]).abs();
            out[y * input.width + x] = ((gx + gy) * 3.0).clamp(0.0, 1.0);
        }
    }
    out
}

fn color_attenuation_transmission_prior(input: &RgbImageF32) -> Vec<f32> {
    input
        .data
        .iter()
        .map(|p| {
            let max_c = p[0].max(p[1]).max(p[2]);
            let min_c = p[0].min(p[1]).min(p[2]);
            let saturation = if max_c > 1e-6 {
                (max_c - min_c) / max_c
            } else {
                0.0
            };
            let brightness = max_c;
            let depth =
                (0.121_779 + 0.959_710 * brightness - 0.780_245 * saturation).clamp(0.0, 1.6);
            (-0.85 * depth).exp().clamp(0.12, 0.96)
        })
        .collect()
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

#[derive(Clone, Debug)]
struct TemporalState {
    transmission: Vec<f32>,
    restored_luma: Vec<f32>,
    source_luma: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DcpTemporalProcessor {
    params: DehazeParams,
    motion: MotionParams,
    temporal: TemporalParams,
    smoother: AirlightSmoother,
    state: Option<TemporalState>,
    frame_index: usize,
}

impl DcpTemporalProcessor {
    pub fn new(
        params: DehazeParams,
        motion: MotionParams,
        temporal: TemporalParams,
        airlight_beta: f32,
        scene_reset_threshold: f32,
    ) -> Self {
        Self {
            params,
            motion,
            temporal,
            smoother: AirlightSmoother::new(airlight_beta, scene_reset_threshold),
            state: None,
            frame_index: 0,
        }
    }

    pub fn process_frame(&mut self, input: &RgbImageF32) -> (DehazeOutput, TemporalDiagnostics) {
        self.frame_index += 1;
        let dark = dark_channel(input, self.params.patch_size);
        let mask = sky_highlight_mask(
            input,
            self.params.sky_protect,
            self.params.highlight_protect,
        );
        let raw_airlight = estimate_airlight_robust(input, &dark, &mask);
        let stats = FrameStats::from_image(input);
        let (airlight, scene_reset) = self.smoother.update_with_stats(raw_airlight, stats);
        let mut output = dehaze_frame_with_airlight_cap_fusion_and_mask(
            input,
            &self.params,
            airlight,
            dark,
            mask,
        );

        let current_luma = input.luma();
        let mut flicker = 0.0;
        let mut mean_motion_error = 0.0;
        let mut warped_transmission_debug = vec![0.0; input.width * input.height];
        let mut motion_magnitude_debug = vec![0.0; input.width * input.height];
        let mut motion_error_debug = vec![0.0; input.width * input.height];

        if !scene_reset {
            if let Some(prev) = &self.state {
                if prev.source_luma.len() == current_luma.len() {
                    let motion = estimate_block_motion(
                        &prev.source_luma,
                        &current_luma,
                        input.width,
                        input.height,
                        &self.motion,
                    );
                    let warped_t = warp_scalar(
                        &prev.transmission,
                        input.width,
                        input.height,
                        &motion.flow_x,
                        &motion.flow_y,
                    );
                    let warped_luma = warp_scalar(
                        &prev.restored_luma,
                        input.width,
                        input.height,
                        &motion.flow_x,
                        &motion.flow_y,
                    );
                    mean_motion_error = mean(&motion.error);
                    flicker = mean_abs_diff_with_mode(
                        &output.image.luma(),
                        &warped_luma,
                        self.params.simd,
                    );
                    warped_transmission_debug.clone_from(&warped_t);
                    motion_error_debug.clone_from(&motion.error);
                    for i in 0..motion_magnitude_debug.len() {
                        motion_magnitude_debug[i] =
                            (motion.flow_x[i].powi(2) + motion.flow_y[i].powi(2)).sqrt();
                    }
                    for i in 0..output.transmission.len() {
                        let residual = (current_luma[i] - warped_luma[i]).abs();
                        let occluded = residual > self.temporal.occlusion_threshold;
                        let confidence = if occluded {
                            0.0
                        } else {
                            let protection = effective_protection(
                                output.protection_mask[i],
                                self.params.aggressiveness,
                            );
                            (1.0 - residual / self.temporal.occlusion_threshold.max(1e-4))
                                .clamp(0.0, 1.0)
                                * (1.0 - protection * 0.35)
                        };
                        let weight = self.temporal.temporal_weight.clamp(0.0, 0.95) * confidence;
                        output.transmission[i] =
                            output.transmission[i] * (1.0 - weight) + warped_t[i] * weight;
                    }
                    output.image = recover_protected(
                        input,
                        output.airlight,
                        &output.transmission,
                        &output.protection_mask,
                        self.params.t_min,
                        self.params.aggressiveness,
                    );
                    output.mean_transmission = mean(&output.transmission);
                }
            }
        }

        self.state = Some(TemporalState {
            transmission: output.transmission.clone(),
            restored_luma: output.image.luma(),
            source_luma: current_luma,
        });

        let diagnostics = TemporalDiagnostics {
            frame_index: self.frame_index,
            airlight,
            raw_airlight,
            mean_transmission: output.mean_transmission,
            flicker,
            mean_motion_error,
            scene_reset,
            debug: Some(TemporalDebug {
                width: input.width,
                height: input.height,
                transmission: output.transmission.clone(),
                warped_transmission: warped_transmission_debug,
                protection_mask: output.protection_mask.clone(),
                motion_magnitude: motion_magnitude_debug,
                motion_error: motion_error_debug,
            }),
        };
        (output, diagnostics)
    }
}

#[derive(Clone, Debug)]
struct MotionField {
    flow_x: Vec<f32>,
    flow_y: Vec<f32>,
    error: Vec<f32>,
}

fn estimate_block_motion(
    prev: &[f32],
    curr: &[f32],
    width: usize,
    height: usize,
    params: &MotionParams,
) -> MotionField {
    let block = params.block_size.max(4);
    let levels = params.pyramid_levels.max(1);
    let radius = params.search_radius.max(1) * levels;
    let mut flow_x = vec![0.0; width * height];
    let mut flow_y = vec![0.0; width * height];
    let mut error = vec![0.0; width * height];

    for by in (0..height).step_by(block) {
        for bx in (0..width).step_by(block) {
            let bw = block.min(width - bx);
            let bh = block.min(height - by);
            let (dx, dy, sad) =
                best_block_offset(prev, curr, width, height, bx, by, bw, bh, radius);
            let norm_error = sad / (bw * bh).max(1) as f32;
            for y in by..(by + bh) {
                for x in bx..(bx + bw) {
                    let idx = y * width + x;
                    flow_x[idx] = dx as f32;
                    flow_y[idx] = dy as f32;
                    error[idx] = norm_error;
                }
            }
        }
    }

    MotionField {
        flow_x,
        flow_y,
        error,
    }
}

fn best_block_offset(
    prev: &[f32],
    curr: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    radius: usize,
) -> (isize, isize, f32) {
    let mut best = (0isize, 0isize, f32::INFINITY);
    let radius = radius as isize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let mut sad = 0.0;
            let mut count = 0usize;
            for y in by..(by + bh) {
                let py = y as isize + dy;
                if py < 0 || py >= height as isize {
                    continue;
                }
                for x in bx..(bx + bw) {
                    let px = x as isize + dx;
                    if px < 0 || px >= width as isize {
                        continue;
                    }
                    sad += (curr[y * width + x] - prev[py as usize * width + px as usize]).abs();
                    count += 1;
                }
            }
            if count > 0 {
                let normalized = sad / count as f32 + 0.001 * ((dx * dx + dy * dy) as f32).sqrt();
                if normalized < best.2 {
                    best = (dx, dy, normalized);
                }
            }
        }
    }
    best
}

fn warp_scalar(
    input: &[f32],
    width: usize,
    height: usize,
    flow_x: &[f32],
    flow_y: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            out[idx] = sample_bilinear(
                input,
                width,
                height,
                x as f32 + flow_x[idx],
                y as f32 + flow_y[idx],
            );
        }
    }
    out
}

fn sample_bilinear(input: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let a = input[y0 * width + x0] * (1.0 - tx) + input[y0 * width + x1] * tx;
    let b = input[y1 * width + x0] * (1.0 - tx) + input[y1 * width + x1] * tx;
    a * (1.0 - ty) + b * ty
}

pub fn mean_abs_diff_with_mode(a: &[f32], b: &[f32], mode: SimdMode) -> f32 {
    assert_eq!(a.len(), b.len());
    if mode.allow_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return mean_abs_diff_avx2(a, b);
        }
    }
    mean_abs_diff_scalar(a, b)
}

fn mean_abs_diff_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len().min(b.len()).max(1) as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mean_abs_diff_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut i = 0usize;
    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    while i + 8 <= len {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(av, bv);
        let abs = _mm256_andnot_ps(sign_mask, diff);
        sum = _mm256_add_ps(sum, abs);
        i += 8;
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    let mut total = lanes.iter().sum::<f32>();
    while i < len {
        total += (a[i] - b[i]).abs();
        i += 1;
    }
    total / len.max(1) as f32
}

fn cpu_supports_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_channel_uses_local_rgb_minimum() {
        let img = RgbImageF32::new(2, 1, vec![[0.8, 0.7, 0.1], [0.4, 0.9, 0.6]]);
        assert_eq!(dark_channel(&img, 1), vec![0.1, 0.4]);
        assert_eq!(dark_channel(&img, 3), vec![0.1, 0.1]);
    }

    #[test]
    fn min_filter_matches_naive_window_minimum() {
        let input = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.2, 0.4];
        let fast = min_filter_gray(4, 3, &input, 1);
        let mut naive = vec![0.0; input.len()];
        for y in 0usize..3 {
            for x in 0usize..4 {
                let mut m = 1.0f32;
                for yy in y.saturating_sub(1)..=(y + 1).min(2) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(3) {
                        m = m.min(input[yy * 4 + xx]);
                    }
                }
                naive[y * 4 + x] = m;
            }
        }
        assert_eq!(fast, naive);
    }

    #[test]
    fn airlight_smoother_respects_beta() {
        let mut smoother = AirlightSmoother::new(0.25, 1.0);
        assert_eq!(smoother.update([1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]);
        let next = smoother.update([0.0, 0.0, 0.0]);
        assert!((next[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn robust_airlight_penalizes_white_highlight_candidates() {
        let img = RgbImageF32::new(
            4,
            1,
            vec![
                [1.0, 1.0, 1.0],
                [0.72, 0.74, 0.76],
                [0.20, 0.20, 0.20],
                [0.10, 0.30, 0.60],
            ],
        );
        let dark = vec![0.99, 0.74, 0.20, 0.10];
        let mask = vec![1.0, 0.0, 0.0, 0.0];
        let airlight = estimate_airlight_robust(&img, &dark, &mask);
        assert_eq!(airlight, [0.72, 0.74, 0.76]);
    }

    #[test]
    fn sky_highlight_mask_detects_bright_low_texture_regions() {
        let img = RgbImageF32::new(
            10,
            1,
            vec![
                [0.67, 0.72, 0.86],
                [0.68, 0.73, 0.87],
                [0.67, 0.72, 0.86],
                [0.68, 0.73, 0.87],
                [0.67, 0.72, 0.86],
                [0.68, 0.73, 0.87],
                [0.67, 0.72, 0.86],
                [0.68, 0.73, 0.87],
                [0.67, 0.72, 0.86],
                [0.10, 0.28, 0.10],
            ],
        );
        let mask = sky_highlight_mask(&img, true, true);
        assert!(mask[0] > 0.0);
        assert!(mask[1] > 0.0);
        assert_eq!(mask[9], 0.0);
    }

    #[test]
    fn guided_filter_keeps_synthetic_edge_sharper_than_box_filter() {
        let guide = RgbImageF32::new(
            6,
            1,
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        );
        let input = vec![0.1, 0.1, 0.1, 0.9, 0.9, 0.9];
        let guided = guided_filter_rgb(&guide, &input, 2, 1e-4);
        let boxed = box_filter_gray(6, 1, &input, 2);
        assert!(guided[2] < boxed[2]);
        assert!(guided[3] > boxed[3]);
    }

    #[test]
    fn block_motion_warp_tracks_translation() {
        let prev = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0,
            0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let curr = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0,
            0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let field = estimate_block_motion(
            &prev,
            &curr,
            5,
            5,
            &MotionParams {
                search_radius: 2,
                block_size: 5,
                pyramid_levels: 1,
            },
        );
        assert!(field.flow_x[12] < -0.5);
    }

    #[test]
    fn scene_reset_uses_histogram_like_stats() {
        let mut smoother = AirlightSmoother::new(0.25, 0.10);
        let a = RgbImageF32::blank(4, 4, [0.1, 0.1, 0.1]);
        let b = RgbImageF32::blank(4, 4, [0.9, 0.9, 0.9]);
        let (_, first_reset) =
            smoother.update_with_stats([0.5, 0.5, 0.5], FrameStats::from_image(&a));
        let (_, second_reset) =
            smoother.update_with_stats([0.55, 0.55, 0.55], FrameStats::from_image(&b));
        assert!(!first_reset);
        assert!(second_reset);
    }

    #[test]
    fn improved_dcp_produces_valid_transmission() {
        let clear = RgbImageF32::new(
            4,
            2,
            vec![
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
                [0.5, 0.6, 0.7],
                [0.6, 0.5, 0.4],
                [0.7, 0.6, 0.5],
                [0.8, 0.7, 0.6],
                [0.9, 0.8, 0.7],
            ],
        );
        let hazy = synthesize_haze(&clear, 1.0, [1.0, 1.0, 1.0]);
        let output = improved_dcp(
            &hazy,
            &DehazeParams {
                patch_size: 3,
                refine_radius: 1,
                ..DehazeParams::default()
            },
        );
        assert_eq!(output.transmission.len(), 8);
        assert!(output
            .transmission
            .iter()
            .all(|&t| (0.0..=1.0).contains(&t)));
    }

    #[test]
    fn original_dcp_produces_valid_image_shape_and_range() {
        let img = RgbImageF32::new(
            3,
            2,
            vec![
                [0.65, 0.68, 0.72],
                [0.45, 0.50, 0.55],
                [0.30, 0.35, 0.40],
                [0.55, 0.53, 0.50],
                [0.20, 0.24, 0.30],
                [0.80, 0.82, 0.85],
            ],
        );
        let output = original_dcp(
            &img,
            &DehazeParams {
                patch_size: 3,
                refine_radius: 1,
                ..DehazeParams::default()
            },
        );
        assert_eq!(output.image.width, img.width);
        assert_eq!(output.image.height, img.height);
        assert!(output
            .image
            .data
            .iter()
            .flat_map(|p| p.iter())
            .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn simd_abs_diff_matches_scalar() {
        let a = (0..31).map(|v| v as f32 * 0.13).collect::<Vec<_>>();
        let b = (0..31).map(|v| 2.0 - v as f32 * 0.07).collect::<Vec<_>>();
        let scalar = mean_abs_diff_with_mode(&a, &b, SimdMode::Scalar);
        let auto = mean_abs_diff_with_mode(&a, &b, SimdMode::Auto);
        assert!((scalar - auto).abs() < 1e-6);
    }

    #[test]
    fn simd_luma_matches_scalar() {
        let data = (0..37)
            .map(|v| {
                [
                    ((v * 13) % 31) as f32 / 30.0,
                    ((v * 17) % 29) as f32 / 28.0,
                    ((v * 19) % 23) as f32 / 22.0,
                ]
            })
            .collect::<Vec<_>>();
        let img = RgbImageF32::new(37, 1, data);
        let scalar = img.luma_with_mode(SimdMode::Scalar);
        let auto = img.luma_with_mode(SimdMode::Auto);
        for (a, b) in scalar.iter().zip(auto.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn simd_transmission_matches_scalar() {
        let dark = (0..37)
            .map(|v| ((v * 17) % 29) as f32 / 28.0)
            .collect::<Vec<_>>();
        let scalar = transmission_from_dark_with_mode(&dark, 0.94, SimdMode::Scalar);
        let auto = transmission_from_dark_with_mode(&dark, 0.94, SimdMode::Auto);
        for (a, b) in scalar.iter().zip(auto.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
