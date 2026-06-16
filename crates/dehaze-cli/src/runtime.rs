use anyhow::{anyhow, bail, Context, Result};
use dehaze_core::{
    color_delta, flicker, improved_dcp, original_dcp, psnr, ssim, synthesize_haze,
    DcpTemporalProcessor, DehazeOutput, DehazeParams, MotionParams, RgbImageF32, TemporalDebug,
    TemporalParams,
};
use image::{ImageBuffer, Rgb};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    OriginalDcp,
    ImprovedDcp,
    Neural,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Cpu,
    PythonCuda,
}

#[derive(Clone, Debug)]
pub struct NeuralConfig {
    pub script: PathBuf,
    pub model: PathBuf,
    pub debug_dir: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct CommonConfig {
    pub method: Method,
    pub backend: Backend,
    pub params: DehazeParams,
    pub neural: NeuralConfig,
}

pub struct ImageConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub common: CommonConfig,
}

pub struct EvalImagesConfig {
    pub hazy_dir: PathBuf,
    pub gt_dir: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub csv: Option<PathBuf>,
    pub max_images: Option<usize>,
    pub max_side: Option<u32>,
    pub common: CommonConfig,
}

pub struct EvalSequencesConfig {
    pub hazy_dir: PathBuf,
    pub gt_dir: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub video_dir: Option<PathBuf>,
    pub csv: Option<PathBuf>,
    pub max_scenes: Option<usize>,
    pub max_frames: Option<usize>,
    pub max_side: Option<u32>,
    pub fps: f32,
    pub common: CommonConfig,
}

pub struct VideoConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub metrics_csv: Option<PathBuf>,
    pub diagnostics_dir: Option<PathBuf>,
    pub diagnostic_previews: bool,
    pub keep_temp: bool,
    pub airlight_beta: f32,
    pub scene_reset_threshold: f32,
    pub motion_search_radius: usize,
    pub motion_block_size: usize,
    pub motion_pyramid_levels: usize,
    pub temporal_weight: f32,
    pub occlusion_threshold: f32,
    pub common: CommonConfig,
}

pub struct MetricsConfig {
    pub pred_dir: PathBuf,
    pub gt_dir: PathBuf,
    pub csv: PathBuf,
}

pub struct SynthesizeConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub beta: f32,
    pub airlight: String,
    pub keep_temp: bool,
}

pub fn process_image(cfg: ImageConfig) -> Result<()> {
    ensure_method_backend(&cfg.common)?;
    let start = Instant::now();
    let output = run_frame(&cfg.input, None, &cfg.common, None)?;
    save_image(&cfg.output, &output.image)?;
    println!(
        "method={} backend={} elapsed_ms={} airlight={:.4},{:.4},{:.4}",
        method_name(cfg.common.method),
        backend_name(cfg.common.backend),
        start.elapsed().as_millis(),
        output.airlight[0],
        output.airlight[1],
        output.airlight[2]
    );
    Ok(())
}

pub fn eval_images(cfg: EvalImagesConfig) -> Result<()> {
    ensure_method_backend(&cfg.common)?;
    let mut pairs = paired_images(&cfg.hazy_dir, &cfg.gt_dir)?;
    if let Some(limit) = cfg.max_images {
        pairs.truncate(limit);
    }
    if pairs.is_empty() {
        bail!(
            "no paired images found under {} and {}",
            cfg.hazy_dir.display(),
            cfg.gt_dir.display()
        );
    }
    if let Some(dir) = &cfg.output_dir {
        fs::create_dir_all(dir)?;
    }

    let mut csv = String::from(
        "name,input_psnr,input_ssim,output_psnr,output_ssim,color_delta,flicker,elapsed_ms,method,backend,width,height\n",
    );
    let mut sums = MetricSums::default();
    for pair in &pairs {
        let hazy = load_image_maybe_resized(&pair.hazy, cfg.max_side)?;
        let gt = load_image_maybe_resized(&pair.gt, cfg.max_side)?;
        ensure_same_dims(&hazy, &gt, &pair.name)?;
        let input_for_neural = if cfg.max_side.is_some() {
            None
        } else {
            Some(pair.hazy.as_path())
        };
        let start = Instant::now();
        let output = run_image_data(&hazy, input_for_neural, &cfg.common, None)?;
        let elapsed_ms = start.elapsed().as_millis();
        if let Some(dir) = &cfg.output_dir {
            save_image(&dir.join(&pair.name), &output.image)?;
        }
        let row = metric_row(
            &gt,
            &hazy,
            &output.image,
            None,
            elapsed_ms,
            cfg.common.method,
            cfg.common.backend,
        );
        sums.add(&row);
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}\n",
            pair.name.display(),
            row.input_psnr,
            row.input_ssim,
            row.output_psnr,
            row.output_ssim,
            row.color_delta,
            row.flicker,
            elapsed_ms,
            method_name(cfg.common.method),
            backend_name(cfg.common.backend),
            hazy.width,
            hazy.height
        ));
    }
    let avg = sums.average(pairs.len());
    csv.push_str(&format!(
        "average,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{},{},,\n",
        avg.input_psnr,
        avg.input_ssim,
        avg.output_psnr,
        avg.output_ssim,
        avg.color_delta,
        avg.flicker,
        avg.elapsed_ms,
        method_name(cfg.common.method),
        backend_name(cfg.common.backend)
    ));
    if let Some(path) = cfg.csv {
        write_text(&path, &csv)?;
    }
    println!(
        "eval-images images={} method={} backend={} input_psnr={:.3} input_ssim={:.4} output_psnr={:.3} output_ssim={:.4} color_delta={:.4} flicker={:.4} elapsed_ms={:.1}",
        pairs.len(),
        method_name(cfg.common.method),
        backend_name(cfg.common.backend),
        avg.input_psnr,
        avg.input_ssim,
        avg.output_psnr,
        avg.output_ssim,
        avg.color_delta,
        avg.flicker,
        avg.elapsed_ms
    );
    Ok(())
}

pub fn eval_sequences(cfg: EvalSequencesConfig) -> Result<()> {
    ensure_method_backend(&cfg.common)?;
    if cfg.video_dir.is_some() && cfg.output_dir.is_none() {
        bail!("--video-dir requires --output-dir");
    }
    if cfg.video_dir.is_some() {
        ensure_command("ffmpeg")?;
    }
    let mut scenes = grouped_image_sequences(&cfg.hazy_dir, &cfg.gt_dir)?;
    if let Some(limit) = cfg.max_scenes {
        scenes.truncate(limit);
    }
    if scenes.is_empty() {
        bail!(
            "no paired frame sequences found under {} and {}",
            cfg.hazy_dir.display(),
            cfg.gt_dir.display()
        );
    }
    if let Some(dir) = &cfg.output_dir {
        fs::create_dir_all(dir)?;
    }
    if let Some(dir) = &cfg.video_dir {
        fs::create_dir_all(dir)?;
    }
    if matches!(cfg.common.method, Method::Neural) {
        return eval_sequences_neural_batch(cfg, scenes);
    }

    let mut csv = String::from(
        "scene,frame,input_psnr,input_ssim,output_psnr,output_ssim,color_delta,flicker,elapsed_ms,method,backend,width,height\n",
    );
    let mut sums = MetricSums::default();
    let mut total_frames = 0usize;
    for (scene, mut frames) in scenes {
        if let Some(limit) = cfg.max_frames {
            frames.truncate(limit);
        }
        let scene_out = cfg.output_dir.as_ref().map(|dir| dir.join(&scene));
        if let Some(dir) = &scene_out {
            fs::create_dir_all(dir)?;
        }
        let mut prev_output: Option<RgbImageF32> = None;
        let mut temporal = make_temporal_processor(&cfg.common, 0.25, 0.18, 6, 12, 2, 0.65, 0.18);
        for (idx, pair) in frames.iter().enumerate() {
            let hazy = load_image_maybe_resized(&pair.hazy, cfg.max_side)?;
            let gt = load_image_maybe_resized(&pair.gt, cfg.max_side)?;
            ensure_same_dims(&hazy, &gt, &pair.name)?;
            let input_for_neural = if cfg.max_side.is_some() {
                None
            } else {
                Some(pair.hazy.as_path())
            };
            let start = Instant::now();
            let output = run_image_data(&hazy, input_for_neural, &cfg.common, Some(&mut temporal))?;
            let elapsed_ms = start.elapsed().as_millis();
            if let Some(dir) = &scene_out {
                save_image(
                    &dir.join(format!("frame_{:06}.png", idx + 1)),
                    &output.image,
                )?;
            }
            let row = metric_row(
                &gt,
                &hazy,
                &output.image,
                prev_output.as_ref(),
                elapsed_ms,
                cfg.common.method,
                cfg.common.backend,
            );
            prev_output = Some(output.image);
            sums.add(&row);
            total_frames += 1;
            csv.push_str(&format!(
                "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}\n",
                scene,
                idx + 1,
                row.input_psnr,
                row.input_ssim,
                row.output_psnr,
                row.output_ssim,
                row.color_delta,
                row.flicker,
                elapsed_ms,
                method_name(cfg.common.method),
                backend_name(cfg.common.backend),
                hazy.width,
                hazy.height
            ));
        }
        if let (Some(output_dir), Some(video_dir)) = (&cfg.output_dir, &cfg.video_dir) {
            encode_frames(
                &output_dir.join(&scene),
                cfg.fps,
                &video_dir.join(format!("{scene}.mp4")),
            )?;
        }
    }
    let avg = sums.average(total_frames);
    if let Some(path) = cfg.csv {
        write_text(&path, &csv)?;
    }
    println!(
        "eval-sequences frames={} method={} backend={} input_psnr={:.3} input_ssim={:.4} output_psnr={:.3} output_ssim={:.4} color_delta={:.4} flicker={:.4} elapsed_ms={:.1}",
        total_frames,
        method_name(cfg.common.method),
        backend_name(cfg.common.backend),
        avg.input_psnr,
        avg.input_ssim,
        avg.output_psnr,
        avg.output_ssim,
        avg.color_delta,
        avg.flicker,
        avg.elapsed_ms
    );
    Ok(())
}

pub fn video(cfg: VideoConfig) -> Result<()> {
    ensure_method_backend(&cfg.common)?;
    ensure_command("ffmpeg")?;
    let fps = probe_fps(&cfg.input).unwrap_or(25.0);
    let temp = TempWork::new("dehaze-video")?;
    let in_dir = temp.path.join("frames_in");
    let out_dir = temp.path.join("frames_out");
    fs::create_dir_all(&in_dir)?;
    fs::create_dir_all(&out_dir)?;
    extract_frames(&cfg.input, &in_dir)?;
    let frames = list_pngs(&in_dir)?;
    if frames.is_empty() {
        bail!("ffmpeg produced no frames for {}", cfg.input.display());
    }
    if matches!(cfg.common.method, Method::Neural) {
        let start = Instant::now();
        run_neural_dir(&in_dir, &out_dir, None, &cfg.common)?;
        let elapsed_ms = start.elapsed().as_millis();
        let per_frame = elapsed_ms as f32 / frames.len().max(1) as f32;
        let mut csv = String::from(
            "frame,airlight_r,airlight_g,airlight_b,mean_transmission,elapsed_ms,method,backend\n",
        );
        for idx in 0..frames.len() {
            csv.push_str(&format!(
                "{},1.000000,1.000000,1.000000,1.000000,{:.3},{},{}\n",
                idx + 1,
                per_frame,
                method_name(cfg.common.method),
                backend_name(cfg.common.backend)
            ));
        }
        encode_frames(&out_dir, fps, &cfg.output)?;
        if let Some(path) = cfg.metrics_csv {
            write_text(&path, &csv)?;
        }
        println!(
            "video frames={} method={} backend={} elapsed_s={:.2} fps={:.2}",
            frames.len(),
            method_name(cfg.common.method),
            backend_name(cfg.common.backend),
            start.elapsed().as_secs_f32(),
            frames.len() as f32 / start.elapsed().as_secs_f32().max(0.001)
        );
        if cfg.keep_temp {
            temp.keep();
        }
        return Ok(());
    }

    let mut temporal = make_temporal_processor(
        &cfg.common,
        cfg.airlight_beta,
        cfg.scene_reset_threshold,
        cfg.motion_search_radius,
        cfg.motion_block_size,
        cfg.motion_pyramid_levels,
        cfg.temporal_weight,
        cfg.occlusion_threshold,
    );
    let mut csv = String::from(
        "frame,airlight_r,airlight_g,airlight_b,mean_transmission,elapsed_ms,method,backend\n",
    );
    let mut diagnostics_csv = String::from(
        "frame,airlight_r,airlight_g,airlight_b,raw_airlight_r,raw_airlight_g,raw_airlight_b,mean_transmission,flicker,mean_motion_error,scene_reset,elapsed_ms\n",
    );
    let start_all = Instant::now();
    for (idx, frame) in frames.iter().enumerate() {
        let start = Instant::now();
        let input = load_image(frame)?;
        let output = if matches!(cfg.common.method, Method::ImprovedDcp) {
            let (output, diag) = temporal.process_frame(&input);
            diagnostics_csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
                diag.frame_index,
                diag.airlight[0],
                diag.airlight[1],
                diag.airlight[2],
                diag.raw_airlight[0],
                diag.raw_airlight[1],
                diag.raw_airlight[2],
                diag.mean_transmission,
                diag.flicker,
                diag.mean_motion_error,
                diag.scene_reset,
                start.elapsed().as_millis()
            ));
            if cfg.diagnostic_previews {
                if let (Some(dir), Some(debug)) = (&cfg.diagnostics_dir, &diag.debug) {
                    save_temporal_previews(dir, diag.frame_index, debug)?;
                }
            }
            output
        } else {
            run_image_data(&input, Some(frame.as_path()), &cfg.common, None)?
        };
        let elapsed_ms = start.elapsed().as_millis();
        save_image(
            &out_dir.join(format!("frame_{:06}.png", idx + 1)),
            &output.image,
        )?;
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
            idx + 1,
            output.airlight[0],
            output.airlight[1],
            output.airlight[2],
            output.mean_transmission,
            elapsed_ms,
            method_name(cfg.common.method),
            backend_name(cfg.common.backend)
        ));
    }
    encode_frames(&out_dir, fps, &cfg.output)?;
    if let Some(path) = cfg.metrics_csv {
        write_text(&path, &csv)?;
    }
    if let Some(dir) = cfg.diagnostics_dir {
        fs::create_dir_all(&dir)?;
        write_text(&dir.join("airlight.csv"), &csv)?;
        write_text(&dir.join("temporal.csv"), &diagnostics_csv)?;
    }
    println!(
        "video frames={} method={} backend={} elapsed_s={:.2} fps={:.2}",
        frames.len(),
        method_name(cfg.common.method),
        backend_name(cfg.common.backend),
        start_all.elapsed().as_secs_f32(),
        frames.len() as f32 / start_all.elapsed().as_secs_f32().max(0.001)
    );
    if cfg.keep_temp {
        temp.keep();
    }
    Ok(())
}

pub fn metrics_dirs(cfg: MetricsConfig) -> Result<()> {
    let pred = list_pngs(&cfg.pred_dir)?;
    let gt = list_pngs(&cfg.gt_dir)?;
    if pred.len() != gt.len() {
        bail!("frame count mismatch: pred={} gt={}", pred.len(), gt.len());
    }
    let mut csv = String::from("frame,output_psnr,output_ssim,color_delta,flicker\n");
    let mut prev: Option<RgbImageF32> = None;
    let mut psnr_sum = 0.0;
    let mut ssim_sum = 0.0;
    let mut delta_sum = 0.0;
    let mut flicker_sum = 0.0;
    for (idx, (pred_path, gt_path)) in pred.iter().zip(gt.iter()).enumerate() {
        let pred_img = load_image(pred_path)?;
        let gt_img = load_image(gt_path)?;
        let p = psnr(&gt_img, &pred_img)
            .ok_or_else(|| anyhow!("dimension mismatch at frame {}", idx + 1))?;
        let s = ssim(&gt_img, &pred_img)
            .ok_or_else(|| anyhow!("dimension mismatch at frame {}", idx + 1))?;
        let d = color_delta(&gt_img, &pred_img).unwrap_or(f32::NAN);
        let f = prev
            .as_ref()
            .and_then(|prev| flicker(prev, &pred_img))
            .unwrap_or(0.0);
        if p.is_finite() {
            psnr_sum += p;
        }
        ssim_sum += s;
        delta_sum += d;
        flicker_sum += f;
        prev = Some(pred_img);
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6}\n",
            idx + 1,
            p,
            s,
            d,
            f
        ));
    }
    let n = pred.len().max(1) as f32;
    csv.push_str(&format!(
        "average,{:.6},{:.6},{:.6},{:.6}\n",
        psnr_sum / n,
        ssim_sum / n,
        delta_sum / n,
        flicker_sum / n
    ));
    write_text(&cfg.csv, &csv)
}

pub fn synthesize_image(cfg: SynthesizeConfig) -> Result<()> {
    let input = load_image(&cfg.input)?;
    let output = synthesize_haze(&input, cfg.beta, parse_airlight(&cfg.airlight)?);
    save_image(&cfg.output, &output)
}

pub fn synthesize_video(cfg: SynthesizeConfig) -> Result<()> {
    ensure_command("ffmpeg")?;
    let fps = probe_fps(&cfg.input).unwrap_or(25.0);
    let temp = TempWork::new("dehaze-synth")?;
    let in_dir = temp.path.join("frames_clear");
    let out_dir = temp.path.join("frames_hazy");
    fs::create_dir_all(&in_dir)?;
    fs::create_dir_all(&out_dir)?;
    extract_frames(&cfg.input, &in_dir)?;
    for (idx, frame) in list_pngs(&in_dir)?.iter().enumerate() {
        let input = load_image(frame)?;
        let output = synthesize_haze(&input, cfg.beta, parse_airlight(&cfg.airlight)?);
        save_image(&out_dir.join(format!("frame_{:06}.png", idx + 1)), &output)?;
    }
    encode_frames(&out_dir, fps, &cfg.output)?;
    if cfg.keep_temp {
        temp.keep();
    }
    Ok(())
}

fn ensure_method_backend(cfg: &CommonConfig) -> Result<()> {
    match (cfg.method, cfg.backend) {
        (Method::OriginalDcp | Method::ImprovedDcp, Backend::Cpu) => Ok(()),
        (Method::OriginalDcp | Method::ImprovedDcp, Backend::PythonCuda) => {
            bail!(
                "{} is a Rust CPU method; use --backend cpu",
                method_name(cfg.method)
            )
        }
        (Method::Neural, Backend::Cpu | Backend::PythonCuda) => Ok(()),
    }
}

fn run_frame(
    path: &Path,
    temporal: Option<&mut DcpTemporalProcessor>,
    cfg: &CommonConfig,
    input: Option<RgbImageF32>,
) -> Result<DehazeOutput> {
    let img = match input {
        Some(img) => img,
        None => load_image(path)?,
    };
    run_image_data(&img, Some(path), cfg, temporal)
}

fn run_image_data(
    input: &RgbImageF32,
    input_path: Option<&Path>,
    cfg: &CommonConfig,
    temporal: Option<&mut DcpTemporalProcessor>,
) -> Result<DehazeOutput> {
    match cfg.method {
        Method::OriginalDcp => Ok(original_dcp(input, &cfg.params)),
        Method::ImprovedDcp => {
            if let Some(temporal) = temporal {
                Ok(temporal.process_frame(input).0)
            } else {
                Ok(improved_dcp(input, &cfg.params))
            }
        }
        Method::Neural => run_neural(input, input_path, cfg),
    }
}

fn run_neural(
    input: &RgbImageF32,
    input_path: Option<&Path>,
    cfg: &CommonConfig,
) -> Result<DehazeOutput> {
    validate_neural_backend(cfg)?;
    let temp;
    let input_file = if let Some(path) = input_path {
        path.to_path_buf()
    } else {
        temp = TempWork::new("dehaze-neural-input")?;
        let path = temp.path.join("input.png");
        fs::create_dir_all(&temp.path)?;
        save_image(&path, input)?;
        path
    };
    let out = TempWork::new("dehaze-neural-output")?;
    fs::create_dir_all(&out.path)?;
    let output_file = out.path.join("output.png");
    let mut cmd = neural_command();
    cmd.arg(&cfg.neural.script)
        .arg("--input")
        .arg(&input_file)
        .arg("--output")
        .arg(&output_file)
        .arg("--model")
        .arg(&cfg.neural.model)
        .arg("--device")
        .arg(match cfg.backend {
            Backend::Cpu => "cpu",
            Backend::PythonCuda => "cuda",
        });
    if let Some(dir) = &cfg.neural.debug_dir {
        cmd.arg("--debug-dir").arg(dir);
    }
    let status = cmd
        .status()
        .with_context(|| "failed to launch Python neural backend")?;
    if !status.success() {
        bail!("Python neural backend failed with status {status}");
    }
    let image = load_image(&output_file)?;
    Ok(DehazeOutput {
        image,
        dark_channel: vec![0.0; input.width * input.height],
        transmission: vec![1.0; input.width * input.height],
        airlight: [1.0, 1.0, 1.0],
        protection_mask: vec![0.0; input.width * input.height],
        mean_transmission: 1.0,
    })
}

fn eval_sequences_neural_batch(
    cfg: EvalSequencesConfig,
    scenes: Vec<(String, Vec<ImagePair>)>,
) -> Result<()> {
    let temp = TempWork::new("dehaze-neural-eval")?;
    let input_dir = temp.path.join("input");
    let output_dir = cfg
        .output_dir
        .clone()
        .unwrap_or_else(|| temp.path.join("output"));
    fs::create_dir_all(&input_dir)?;
    fs::create_dir_all(&output_dir)?;

    let mut selected = Vec::<(String, usize, ImagePair)>::new();
    for (scene, mut frames) in scenes {
        if let Some(limit) = cfg.max_frames {
            frames.truncate(limit);
        }
        for (idx, pair) in frames.into_iter().enumerate() {
            let dst = input_dir.join(&pair.name);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&pair.hazy, &dst).with_context(|| {
                format!(
                    "failed to stage neural input {} -> {}",
                    pair.hazy.display(),
                    dst.display()
                )
            })?;
            selected.push((scene.clone(), idx + 1, pair));
        }
    }
    if selected.is_empty() {
        bail!("no frames selected for neural evaluation");
    }

    let infer_start = Instant::now();
    run_neural_dir(&input_dir, &output_dir, cfg.max_side, &cfg.common)?;
    let per_frame_ms = infer_start.elapsed().as_millis() as f32 / selected.len().max(1) as f32;

    let mut csv = String::from(
        "scene,frame,input_psnr,input_ssim,output_psnr,output_ssim,color_delta,flicker,elapsed_ms,method,backend,width,height\n",
    );
    let mut sums = MetricSums::default();
    let mut prev_by_scene = std::collections::BTreeMap::<String, RgbImageF32>::new();
    for (scene, frame_idx, pair) in &selected {
        let hazy = load_image_maybe_resized(&pair.hazy, cfg.max_side)?;
        let gt = load_image_maybe_resized(&pair.gt, cfg.max_side)?;
        let output_path = output_dir.join(pair.name.with_extension("png"));
        let output = load_image(&output_path)?;
        ensure_same_dims(&hazy, &gt, &pair.name)?;
        ensure_same_dims(&output, &gt, &pair.name)?;
        let row = metric_row(
            &gt,
            &hazy,
            &output,
            prev_by_scene.get(scene),
            per_frame_ms as u128,
            cfg.common.method,
            cfg.common.backend,
        );
        prev_by_scene.insert(scene.clone(), output);
        sums.add(&row);
        csv.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{},{},{},{}\n",
            scene,
            frame_idx,
            row.input_psnr,
            row.input_ssim,
            row.output_psnr,
            row.output_ssim,
            row.color_delta,
            row.flicker,
            per_frame_ms,
            method_name(cfg.common.method),
            backend_name(cfg.common.backend),
            gt.width,
            gt.height
        ));
    }
    let avg = sums.average(selected.len());
    csv.push_str(&format!(
        "average,,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{},{},,\n",
        avg.input_psnr,
        avg.input_ssim,
        avg.output_psnr,
        avg.output_ssim,
        avg.color_delta,
        avg.flicker,
        avg.elapsed_ms,
        method_name(cfg.common.method),
        backend_name(cfg.common.backend)
    ));
    if let Some(path) = cfg.csv {
        write_text(&path, &csv)?;
    }
    if let (Some(output_root), Some(video_dir)) = (&cfg.output_dir, &cfg.video_dir) {
        let mut scenes = selected
            .iter()
            .map(|(scene, _, _)| scene.clone())
            .collect::<Vec<_>>();
        scenes.sort();
        scenes.dedup();
        for scene in scenes {
            encode_frames(
                output_root.join(&scene).as_path(),
                cfg.fps,
                &video_dir.join(format!("{scene}.mp4")),
            )?;
        }
    }
    println!(
        "eval-sequences frames={} method={} backend={} input_psnr={:.3} input_ssim={:.4} output_psnr={:.3} output_ssim={:.4} color_delta={:.4} flicker={:.4} elapsed_ms={:.1}",
        selected.len(),
        method_name(cfg.common.method),
        backend_name(cfg.common.backend),
        avg.input_psnr,
        avg.input_ssim,
        avg.output_psnr,
        avg.output_ssim,
        avg.color_delta,
        avg.flicker,
        avg.elapsed_ms
    );
    Ok(())
}

fn run_neural_dir(
    input_dir: &Path,
    output_dir: &Path,
    max_side: Option<u32>,
    cfg: &CommonConfig,
) -> Result<()> {
    validate_neural_backend(cfg)?;
    let mut cmd = neural_command();
    cmd.arg(&cfg.neural.script)
        .arg("--input-dir")
        .arg(input_dir)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--model")
        .arg(&cfg.neural.model)
        .arg("--device")
        .arg(match cfg.backend {
            Backend::Cpu => "cpu",
            Backend::PythonCuda => "cuda",
        });
    if let Some(max_side) = max_side {
        cmd.arg("--max-side").arg(max_side.to_string());
    }
    let status = cmd
        .status()
        .with_context(|| "failed to launch Python neural batch backend")?;
    if !status.success() {
        bail!("Python neural batch backend failed with status {status}");
    }
    Ok(())
}

fn validate_neural_backend(cfg: &CommonConfig) -> Result<()> {
    if !cfg.neural.script.exists() {
        bail!(
            "neural inference script not found: {}. Expected scripts/neural/infer_neural_dehazer.py",
            cfg.neural.script.display()
        );
    }
    if !cfg.neural.model.exists() {
        bail!(
            "neural model not found: {}. Train it with scripts/neural/train_neural_dehazer.py or pass --model <checkpoint>",
            cfg.neural.model.display()
        );
    }
    Ok(())
}

fn neural_command() -> Command {
    let venv_python = Path::new(".venv/bin/python");
    if venv_python.exists() {
        Command::new(venv_python)
    } else {
        Command::new("python3")
    }
}

fn make_temporal_processor(
    cfg: &CommonConfig,
    airlight_beta: f32,
    scene_reset_threshold: f32,
    search_radius: usize,
    block_size: usize,
    pyramid_levels: usize,
    temporal_weight: f32,
    occlusion_threshold: f32,
) -> DcpTemporalProcessor {
    DcpTemporalProcessor::new(
        cfg.params.clone(),
        MotionParams {
            search_radius,
            block_size,
            pyramid_levels,
        },
        TemporalParams {
            temporal_weight,
            occlusion_threshold,
        },
        airlight_beta,
        scene_reset_threshold,
    )
}

#[derive(Default)]
struct MetricSums {
    input_psnr: f32,
    input_ssim: f32,
    output_psnr: f32,
    output_ssim: f32,
    color_delta: f32,
    flicker: f32,
    elapsed_ms: f32,
}

#[derive(Clone, Copy)]
struct MetricRow {
    input_psnr: f32,
    input_ssim: f32,
    output_psnr: f32,
    output_ssim: f32,
    color_delta: f32,
    flicker: f32,
    elapsed_ms: f32,
}

impl MetricSums {
    fn add(&mut self, row: &MetricRow) {
        self.input_psnr += finite_or_zero(row.input_psnr);
        self.input_ssim += row.input_ssim;
        self.output_psnr += finite_or_zero(row.output_psnr);
        self.output_ssim += row.output_ssim;
        self.color_delta += row.color_delta;
        self.flicker += row.flicker;
        self.elapsed_ms += row.elapsed_ms;
    }

    fn average(&self, n: usize) -> MetricRow {
        let n = n.max(1) as f32;
        MetricRow {
            input_psnr: self.input_psnr / n,
            input_ssim: self.input_ssim / n,
            output_psnr: self.output_psnr / n,
            output_ssim: self.output_ssim / n,
            color_delta: self.color_delta / n,
            flicker: self.flicker / n,
            elapsed_ms: self.elapsed_ms / n,
        }
    }
}

fn metric_row(
    gt: &RgbImageF32,
    hazy: &RgbImageF32,
    output: &RgbImageF32,
    prev_output: Option<&RgbImageF32>,
    elapsed_ms: u128,
    _method: Method,
    _backend: Backend,
) -> MetricRow {
    MetricRow {
        input_psnr: psnr(gt, hazy).unwrap_or(f32::NAN),
        input_ssim: ssim(gt, hazy).unwrap_or(f32::NAN),
        output_psnr: psnr(gt, output).unwrap_or(f32::NAN),
        output_ssim: ssim(gt, output).unwrap_or(f32::NAN),
        color_delta: color_delta(gt, output).unwrap_or(f32::NAN),
        flicker: prev_output
            .and_then(|prev| flicker(prev, output))
            .unwrap_or(0.0),
        elapsed_ms: elapsed_ms as f32,
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn load_image(path: &Path) -> Result<RgbImageF32> {
    let img = image::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .to_rgb8();
    let (width, height) = img.dimensions();
    let data = img
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    Ok(RgbImageF32::new(width as usize, height as usize, data))
}

fn load_image_maybe_resized(path: &Path, max_side: Option<u32>) -> Result<RgbImageF32> {
    let mut img = image::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .to_rgb8();
    if let Some(max_side) = max_side {
        let (w, h) = img.dimensions();
        let current_max = w.max(h);
        if current_max > max_side {
            let scale = max_side as f32 / current_max as f32;
            let new_w = ((w as f32 * scale).round() as u32).max(1);
            let new_h = ((h as f32 * scale).round() as u32).max(1);
            img =
                image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);
        }
    }
    let (width, height) = img.dimensions();
    let data = img
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    Ok(RgbImageF32::new(width as usize, height as usize, data))
}

fn save_image(path: &Path, img: &RgbImageF32) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(img.width as u32, img.height as u32);
    for y in 0..img.height {
        for x in 0..img.width {
            let p = img.get(x, y);
            buffer.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    (p[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (p[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (p[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                ]),
            );
        }
    }
    buffer.save(path)?;
    Ok(())
}

#[derive(Clone)]
struct ImagePair {
    name: PathBuf,
    hazy: PathBuf,
    gt: PathBuf,
}

fn paired_images(hazy_dir: &Path, gt_dir: &Path) -> Result<Vec<ImagePair>> {
    let mut pairs = Vec::new();
    for hazy in collect_images(hazy_dir)? {
        let rel = hazy.strip_prefix(hazy_dir)?.to_path_buf();
        let gt = gt_dir.join(&rel);
        if gt.exists() {
            pairs.push(ImagePair {
                name: rel,
                hazy,
                gt,
            });
        }
    }
    pairs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(pairs)
}

fn grouped_image_sequences(
    hazy_dir: &Path,
    gt_dir: &Path,
) -> Result<Vec<(String, Vec<ImagePair>)>> {
    let mut grouped = std::collections::BTreeMap::<String, Vec<ImagePair>>::new();
    for pair in paired_images(hazy_dir, gt_dir)? {
        let scene = pair
            .name
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or("root")
            .to_string();
        grouped.entry(scene).or_default().push(pair);
    }
    Ok(grouped.into_iter().collect())
}

fn collect_images(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_images_inner(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_images_inner(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_images_inner(&path, out)?;
        } else if is_image(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(OsStr::to_str).map(|s| s.to_ascii_lowercase()),
        Some(ext) if matches!(ext.as_str(), "png" | "jpg" | "jpeg")
    )
}

fn list_pngs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(OsStr::to_str)
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn ensure_same_dims(a: &RgbImageF32, b: &RgbImageF32, name: &Path) -> Result<()> {
    if a.width != b.width || a.height != b.height {
        bail!(
            "dimension mismatch for {}: hazy={}x{} gt={}x{}",
            name.display(),
            a.width,
            a.height,
            b.width,
            b.height
        );
    }
    Ok(())
}

fn ensure_command(name: &str) -> Result<()> {
    let status = Command::new(name).arg("-version").status();
    match status {
        Ok(_) => Ok(()),
        Err(err) => bail!("required command `{name}` is not available: {err}"),
    }
}

fn extract_frames(input: &Path, out_dir: &Path) -> Result<()> {
    let pattern = out_dir.join("frame_%06d.png");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg(pattern)
        .status()
        .with_context(|| "failed to run ffmpeg for frame extraction")?;
    if !status.success() {
        bail!("ffmpeg frame extraction failed for {}", input.display());
    }
    Ok(())
}

fn encode_frames(input_dir: &Path, fps: f32, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let pattern = input_dir.join("frame_%06d.png");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg(format!("{fps:.3}"))
        .arg("-i")
        .arg(pattern)
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(output)
        .status()
        .with_context(|| "failed to run ffmpeg for video encoding")?;
    if !status.success() {
        bail!("ffmpeg video encoding failed for {}", output.display());
    }
    Ok(())
}

fn probe_fps(input: &Path) -> Option<f32> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
        ])
        .arg(input)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let value = text.trim();
    if let Some((num, den)) = value.split_once('/') {
        let num: f32 = num.parse().ok()?;
        let den: f32 = den.parse().ok()?;
        Some(num / den.max(1.0))
    } else {
        value.parse().ok()
    }
}

fn save_temporal_previews(dir: &Path, frame_index: usize, debug: &TemporalDebug) -> Result<()> {
    save_gray_preview(
        &dir.join("transmission"),
        frame_index,
        debug.width,
        debug.height,
        &debug.transmission,
    )?;
    save_gray_preview(
        &dir.join("warped_transmission"),
        frame_index,
        debug.width,
        debug.height,
        &debug.warped_transmission,
    )?;
    save_gray_preview(
        &dir.join("protection_mask"),
        frame_index,
        debug.width,
        debug.height,
        &debug.protection_mask,
    )?;
    save_gray_preview(
        &dir.join("motion_magnitude"),
        frame_index,
        debug.width,
        debug.height,
        &debug.motion_magnitude,
    )?;
    save_gray_preview(
        &dir.join("motion_error"),
        frame_index,
        debug.width,
        debug.height,
        &debug.motion_error,
    )
}

fn save_gray_preview(
    dir: &Path,
    frame_index: usize,
    width: usize,
    height: usize,
    data: &[f32],
) -> Result<()> {
    fs::create_dir_all(dir)?;
    let max_value = data.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    let img = RgbImageF32::new(
        width,
        height,
        data.iter()
            .map(|v| {
                let g = (*v / max_value).clamp(0.0, 1.0);
                [g, g, g]
            })
            .collect(),
    );
    save_image(&dir.join(format!("frame_{frame_index:06}.png")), &img)
}

fn parse_airlight(value: &str) -> Result<[f32; 3]> {
    let parts = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        bail!("airlight must be r,g,b");
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn method_name(method: Method) -> &'static str {
    match method {
        Method::OriginalDcp => "original-dcp",
        Method::ImprovedDcp => "improved-dcp",
        Method::Neural => "neural",
    }
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::PythonCuda => "python-cuda",
    }
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, text)?;
    Ok(())
}

struct TempWork {
    path: PathBuf,
    keep: bool,
}

impl TempWork {
    fn new(prefix: &str) -> Result<Self> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
        fs::create_dir_all(&path)?;
        Ok(Self { path, keep: false })
    }

    fn keep(mut self) {
        self.keep = true;
    }
}

impl Drop for TempWork {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dehaze_core::SimdMode;

    #[test]
    fn neural_backend_reports_missing_model_clearly() {
        let temp = TempWork::new("dehaze-neural-test").unwrap();
        let input = temp.path.join("input.png");
        save_image(&input, &RgbImageF32::blank(2, 2, [0.5, 0.5, 0.5])).unwrap();
        let cfg = CommonConfig {
            method: Method::Neural,
            backend: Backend::Cpu,
            params: DehazeParams {
                simd: SimdMode::Scalar,
                ..DehazeParams::default()
            },
            neural: NeuralConfig {
                script: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../scripts/neural/infer_neural_dehazer.py"),
                model: temp.path.join("missing.pt"),
                debug_dir: None,
            },
        };
        let err = process_image(ImageConfig {
            input,
            output: temp.path.join("out.png"),
            common: cfg,
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("neural model not found"));
    }
}
