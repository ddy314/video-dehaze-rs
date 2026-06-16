mod runtime;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use dehaze_core::{DehazeParams, SimdMode};
use runtime::{
    eval_images, eval_sequences, metrics_dirs, process_image, synthesize_image, synthesize_video,
    video, Backend, EvalImagesConfig, EvalSequencesConfig, ImageConfig, MetricsConfig,
    NeuralConfig, SynthesizeConfig, VideoConfig,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dehaze")]
#[command(about = "Rust DCP and DCP-inspired neural image/video dehazing")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Image(ImageArgs),
    EvalImages(EvalImagesArgs),
    EvalSequences(EvalSequencesArgs),
    Video(VideoArgs),
    Metrics(MetricsArgs),
    SynthesizeImage(SynthesizeImageArgs),
    Synthesize(SynthesizeArgs),
}

#[derive(Parser)]
struct ImageArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Parser)]
struct EvalImagesArgs {
    #[arg(long)]
    hazy_dir: PathBuf,
    #[arg(long)]
    gt_dir: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long)]
    csv: Option<PathBuf>,
    #[arg(long)]
    max_images: Option<usize>,
    #[arg(long)]
    max_side: Option<u32>,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Parser)]
struct EvalSequencesArgs {
    #[arg(long)]
    hazy_dir: PathBuf,
    #[arg(long)]
    gt_dir: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long)]
    video_dir: Option<PathBuf>,
    #[arg(long)]
    csv: Option<PathBuf>,
    #[arg(long)]
    max_scenes: Option<usize>,
    #[arg(long)]
    max_frames: Option<usize>,
    #[arg(long)]
    max_side: Option<u32>,
    #[arg(long, default_value_t = 25.0)]
    fps: f32,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Parser)]
struct VideoArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    metrics_csv: Option<PathBuf>,
    #[arg(long)]
    diagnostics_dir: Option<PathBuf>,
    #[arg(long)]
    diagnostic_previews: bool,
    #[arg(long)]
    keep_temp: bool,
    #[arg(long, default_value_t = 0.25)]
    airlight_beta: f32,
    #[arg(long, default_value_t = 0.18)]
    scene_reset_threshold: f32,
    #[arg(long, default_value_t = 6)]
    motion_search_radius: usize,
    #[arg(long, default_value_t = 12)]
    motion_block_size: usize,
    #[arg(long, default_value_t = 2)]
    motion_pyramid_levels: usize,
    #[arg(long, default_value_t = 0.65)]
    temporal_weight: f32,
    #[arg(long, default_value_t = 0.18)]
    occlusion_threshold: f32,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Parser)]
struct MetricsArgs {
    #[arg(long)]
    pred_dir: PathBuf,
    #[arg(long)]
    gt_dir: PathBuf,
    #[arg(long)]
    csv: PathBuf,
}

#[derive(Parser)]
struct SynthesizeImageArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1.2)]
    beta: f32,
    #[arg(long, default_value = "1.0,1.0,1.0")]
    airlight: String,
}

#[derive(Parser)]
struct SynthesizeArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1.2)]
    beta: f32,
    #[arg(long, default_value = "1.0,1.0,1.0")]
    airlight: String,
    #[arg(long)]
    keep_temp: bool,
}

#[derive(Parser, Clone)]
struct CommonArgs {
    #[arg(long, value_enum, default_value_t = Method::ImprovedDcp)]
    method: Method,
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    backend: BackendArg,
    #[arg(
        long,
        alias = "neural-backend",
        default_value = "scripts/neural/infer_neural_dehazer.py"
    )]
    neural_script: PathBuf,
    #[arg(long, default_value = "models/neural_dehazer.pt")]
    model: PathBuf,
    #[arg(long)]
    debug_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Preset::Balanced)]
    preset: Preset,
    #[arg(long, value_enum, default_value_t = SimdArg::Auto)]
    simd: SimdArg,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    sky_protect: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    highlight_protect: bool,
    #[command(flatten)]
    params: ParamArgs,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Method {
    OriginalDcp,
    ImprovedDcp,
    Neural,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BackendArg {
    Cpu,
    Gpu,
    PythonCuda,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preset {
    Fast,
    Balanced,
    Quality,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SimdArg {
    Auto,
    Scalar,
    Avx2,
}

#[derive(Parser, Clone, Debug)]
struct ParamArgs {
    #[arg(long, default_value_t = 15)]
    patch_size: usize,
    #[arg(long, default_value_t = 0.96)]
    omega: f32,
    #[arg(long, default_value_t = 0.10)]
    t_min: f32,
    #[arg(long, default_value_t = 12)]
    refine_radius: usize,
    #[arg(long, default_value_t = 0.001)]
    guided_eps: f32,
    #[arg(long, default_value_t = 0.20)]
    aggressiveness: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Image(args) => process_image(ImageConfig {
            input: args.input,
            output: args.output,
            common: args.common.into(),
        }),
        Commands::EvalImages(args) => eval_images(EvalImagesConfig {
            hazy_dir: args.hazy_dir,
            gt_dir: args.gt_dir,
            output_dir: args.output_dir,
            csv: args.csv,
            max_images: args.max_images,
            max_side: args.max_side,
            common: args.common.into(),
        }),
        Commands::EvalSequences(args) => eval_sequences(EvalSequencesConfig {
            hazy_dir: args.hazy_dir,
            gt_dir: args.gt_dir,
            output_dir: args.output_dir,
            video_dir: args.video_dir,
            csv: args.csv,
            max_scenes: args.max_scenes,
            max_frames: args.max_frames,
            max_side: args.max_side,
            fps: args.fps,
            common: args.common.into(),
        }),
        Commands::Video(args) => video(VideoConfig {
            input: args.input,
            output: args.output,
            metrics_csv: args.metrics_csv,
            diagnostics_dir: args.diagnostics_dir,
            diagnostic_previews: args.diagnostic_previews,
            keep_temp: args.keep_temp,
            airlight_beta: args.airlight_beta,
            scene_reset_threshold: args.scene_reset_threshold,
            motion_search_radius: args.motion_search_radius,
            motion_block_size: args.motion_block_size,
            motion_pyramid_levels: args.motion_pyramid_levels,
            temporal_weight: args.temporal_weight,
            occlusion_threshold: args.occlusion_threshold,
            common: args.common.into(),
        }),
        Commands::Metrics(args) => metrics_dirs(MetricsConfig {
            pred_dir: args.pred_dir,
            gt_dir: args.gt_dir,
            csv: args.csv,
        }),
        Commands::SynthesizeImage(args) => synthesize_image(SynthesizeConfig {
            input: args.input,
            output: args.output,
            beta: args.beta,
            airlight: args.airlight,
            keep_temp: false,
        }),
        Commands::Synthesize(args) => synthesize_video(SynthesizeConfig {
            input: args.input,
            output: args.output,
            beta: args.beta,
            airlight: args.airlight,
            keep_temp: args.keep_temp,
        }),
    }
}

impl From<CommonArgs> for runtime::CommonConfig {
    fn from(args: CommonArgs) -> Self {
        let mut params = DehazeParams {
            patch_size: args.params.patch_size,
            omega: args.params.omega,
            t_min: args.params.t_min,
            refine_radius: preset_refine_radius(args.preset, args.params.refine_radius),
            guided_eps: args.params.guided_eps,
            aggressiveness: args.params.aggressiveness.clamp(0.0, 1.0),
            sky_protect: args.sky_protect,
            highlight_protect: args.highlight_protect,
            simd: match args.simd {
                SimdArg::Auto => SimdMode::Auto,
                SimdArg::Scalar => SimdMode::Scalar,
                SimdArg::Avx2 => SimdMode::Avx2,
            },
        };
        if matches!(args.method, Method::OriginalDcp) {
            params.sky_protect = false;
            params.highlight_protect = false;
            params.aggressiveness = 0.0;
        }
        runtime::CommonConfig {
            method: match args.method {
                Method::OriginalDcp => runtime::Method::OriginalDcp,
                Method::ImprovedDcp => runtime::Method::ImprovedDcp,
                Method::Neural => runtime::Method::Neural,
            },
            backend: match args.backend {
                BackendArg::Cpu => Backend::Cpu,
                BackendArg::Gpu | BackendArg::PythonCuda => Backend::PythonCuda,
            },
            params,
            neural: NeuralConfig {
                script: args.neural_script,
                model: args.model,
                debug_dir: args.debug_dir,
            },
        }
    }
}

fn preset_refine_radius(preset: Preset, value: usize) -> usize {
    match preset {
        Preset::Fast => value.min(6),
        Preset::Balanced => value,
        Preset::Quality => value.max(16),
    }
}
