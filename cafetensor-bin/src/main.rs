use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};

use cafetensor_lib::container::{self, EXTENSION};
use cafetensor_lib::safetensors::Source;
use cafetensor_lib::tensor::{
    MAX_BLOCK_LOG2, MIN_BLOCK_LOG2, Options, compress_bytes, decompress_tensor, is_passthrough,
};
use cafetensor_lib::{HugeBuf, SmMode, hash, shards};

#[derive(Parser)]
#[command(
    name = "cafetensor",
    version,
    about = "Lossless rANS compression of safetensors checkpoints"
)]
struct Cli {
    /// Worker threads (default: every core).
    #[arg(short = 'j', long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(clap::Args)]
struct EncodeArgs {
    /// Also entropy code the sign|mantissa byte of BF16/F32: ~0.5% smaller, ~3x slower decode.
    #[arg(long)]
    slow: bool,
    /// Omit the per-chunk CRC32C checksums.
    #[arg(long)]
    no_crc: bool,
    /// Uncompressed bytes per chunk as a power of two (21 = 2 MiB .. 28 = 256 MiB).
    #[arg(long, default_value_t = MIN_BLOCK_LOG2, value_parser = clap::value_parser!(u32).range(MIN_BLOCK_LOG2 as i64..=MAX_BLOCK_LOG2 as i64))]
    block: u32,
}

impl EncodeArgs {
    fn options(&self) -> Options {
        let mode = if self.slow {
            SmMode::Coded
        } else {
            SmMode::Raw
        };
        Options {
            crc: !self.no_crc,
            ..Options::new(1 << self.block, mode)
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Gather one or more .safetensors files into a single .cafetensor.
    Compress {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output file (default: the single input with a .cafetensor extension).
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[command(flatten)]
        enc: EncodeArgs,
        /// Skip decoding every tensor again to compare it before writing.
        #[arg(long)]
        no_verify: bool,
    },
    /// Restore the .safetensors files gathered in a .cafetensor.
    Decompress {
        input: PathBuf,
        /// Output file for a single-source container, otherwise a directory (default: current directory).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Check every restored file against its recorded BLAKE3.
        #[arg(long)]
        verify: bool,
        /// Replace existing files.
        #[arg(long)]
        force: bool,
    },
    /// Print the container header and per-source sizes.
    Inspect { input: PathBuf },
    /// BLAKE3 of files, in the same notation the container records.
    B3sum {
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
    /// Merge sharded .safetensors files into one.
    Join {
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Split one .safetensors file into shards with an index.json.
    Split {
        input: PathBuf,
        /// Output directory.
        #[arg(short, long)]
        output: PathBuf,
        /// Largest shard, e.g. 5GB, 4GiB or a byte count.
        #[arg(long, default_value = "5GB", value_parser = parse_size)]
        max_size: u64,
        /// Shard file name prefix.
        #[arg(long, default_value = "model")]
        prefix: String,
    },
    /// Per-tensor ratio and codec speed of a .safetensors file, in memory.
    Bench {
        input: PathBuf,
        #[command(flatten)]
        enc: EncodeArgs,
    },
}

fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let split = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let num: f64 = num.parse().map_err(|_| format!("bad size {s}"))?;
    let mul = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "kb" => 1e3,
        "mb" => 1e6,
        "gb" => 1e9,
        "tb" => 1e12,
        "kib" => 1024.0,
        "mib" => 1024.0 * 1024.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        u => return Err(format!("unknown size unit {u}")),
    };
    Ok((num * mul) as u64)
}

type Res = Result<(), Box<dyn std::error::Error>>;

fn gbps(bytes: u64, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    eprintln!("cafetensor: tier {}", cafetensor_lib::tier_name());
    if let Some(n) = cli.threads
        && let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
    {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let r = match cli.cmd {
        None => cafetensor_lib::self_test().map_err(Into::into),
        Some(Cmd::Compress {
            inputs,
            output,
            enc,
            no_verify,
        }) => compress(&inputs, output, &enc.options(), !no_verify),
        Some(Cmd::Decompress {
            input,
            output,
            verify,
            force,
        }) => decompress(&input, output, verify, force),
        Some(Cmd::Inspect { input }) => inspect(&input),
        Some(Cmd::B3sum { files }) => b3sum(&files),
        Some(Cmd::Join { inputs, output }) => shards::join(&inputs, &output)
            .map(|n| println!("{n} tensors -> {}", output.display()))
            .map_err(Into::into),
        Some(Cmd::Split {
            input,
            output,
            max_size,
            prefix,
        }) => shards::split(&input, &output, max_size, &prefix)
            .map(|p| println!("{} shards -> {}", p.len(), output.display()))
            .map_err(Into::into),
        Some(Cmd::Bench { input, enc }) => bench(&input, &enc.options()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn compress(inputs: &[PathBuf], output: Option<PathBuf>, opts: &Options, verify: bool) -> Res {
    let output = match (output, inputs) {
        (Some(o), _) => o,
        (None, [one]) => one.with_extension(EXTENSION),
        (None, _) => return Err("several inputs need --output".into()),
    };
    let start = Instant::now();
    let s = container::compress(inputs, &output, opts, verify)?;
    let wall = start.elapsed().as_secs_f64();
    println!(
        "{} sources, {} tensors, {} -> {} bytes ({:.4}x, {:.3} bits/weight), {:.1}s, encode {:.3} GB/s{}",
        inputs.len(),
        s.tensors,
        s.raw_bytes,
        s.packed_bytes,
        s.packed_bytes as f64 / s.raw_bytes.max(1) as f64,
        s.packed_bytes as f64 * 8.0 / s.elements.max(1) as f64,
        wall,
        gbps(s.raw_bytes, s.codec_time.as_secs_f64()),
        if verify { ", verified" } else { "" },
    );
    println!("{} -> {}", s.checksum, output.display());
    Ok(())
}

fn decompress(input: &Path, output: Option<PathBuf>, verify: bool, force: bool) -> Res {
    let output = output.unwrap_or_else(|| PathBuf::from("."));
    let start = Instant::now();
    let s = container::decompress(input, &output, verify, force)?;
    let wall = start.elapsed().as_secs_f64();
    println!(
        "{} tensors, {} bytes restored, {:.1}s, decode {:.3} GB/s ({}, {:?} pages)",
        s.tensors,
        s.raw_bytes,
        wall,
        gbps(s.raw_bytes, s.codec_time.as_secs_f64()),
        cafetensor_lib::tier_name(),
        s.backing
    );
    if verify {
        println!(
            "{} verified, hash {:.3} GB/s",
            s.checksum,
            gbps(s.raw_bytes, s.hash_time.as_secs_f64())
        );
    }
    Ok(())
}

fn inspect(input: &Path) -> Res {
    let mut f = File::open(input)?;
    let h = container::read_header(&mut f)?;
    let m = &h.meta;
    println!("version  {}", m.version);
    println!("block    {} ({} MiB)", m.block, (1u64 << m.block) >> 20);
    println!("crc32c   {}", m.crc32c);
    println!("checksum {}", m.checksum);
    for (i, s) in m.sources.iter().enumerate() {
        let entries = || h.tensors.values().filter(|e| e.source == i);
        let packed: u64 = entries()
            .map(|e| e.data_offsets[1] - e.data_offsets[0])
            .sum();
        println!(
            "source {i} {}  {} tensors  {} -> {} bytes ({:.4}x)  {}",
            s.name,
            entries().count(),
            s.bytes,
            packed,
            packed as f64 / s.bytes.max(1) as f64,
            s.checksum
        );
    }
    let mut passthrough = 0;
    for e in h.tensors.values() {
        let mut head = [0u8; 18];
        f.read_exact_at(&mut head, h.data_start + e.data_offsets[0])?;
        passthrough += usize::from(is_passthrough(&head).unwrap_or(false));
    }
    println!("passthrough tensors {passthrough}");
    Ok(())
}

fn b3sum(files: &[PathBuf]) -> Res {
    for f in files {
        println!("{}  {}", hash::b3sum(f)?, f.display());
    }
    Ok(())
}

fn bench(input: &Path, opts: &Options) -> Res {
    let src = Source::open(input)?;
    let file = File::open(input)?;
    let (mut raw, mut packed_total) = (0u64, 0u64);
    let (mut t_enc, mut t_dec) = (0f64, 0f64);
    let mut buf = HugeBuf::new();
    let mut bytes = Vec::new();
    for t in &src.tensors {
        bytes.resize(t.len(), 0);
        file.read_exact_at(&mut bytes, src.offset(t))?;
        let s = Instant::now();
        let packed = compress_bytes(t.dtype, &bytes, opts)?;
        t_enc += s.elapsed().as_secs_f64();
        let out = buf.resize(bytes.len())?;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let s = Instant::now();
            decompress_tensor(&packed, out)?;
            best = best.min(s.elapsed().as_secs_f64());
        }
        t_dec += best;
        if out != &bytes[..] {
            return Err(format!("mismatch in {}", t.name).into());
        }
        raw += bytes.len() as u64;
        packed_total += packed.len() as u64;
        println!(
            "{:60} {:>8} {:>12} {:.4}x",
            t.name,
            t.dtype.name(),
            t.len(),
            packed.len() as f64 / t.len().max(1) as f64
        );
    }
    println!();
    println!("raw         {raw} bytes");
    println!(
        "compressed  {packed_total} bytes, {:.4}x of raw",
        packed_total as f64 / raw.max(1) as f64
    );
    println!("encode      {:.3} GB/s", gbps(raw, t_enc));
    println!(
        "decode      {:.3} GB/s ({}, {:?} sign|mantissa, crc={})",
        gbps(raw, t_dec),
        cafetensor_lib::tier_name(),
        opts.mode,
        opts.crc
    );
    Ok(())
}
