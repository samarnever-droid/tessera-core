//! `axiom-bench`: Benchmark harness library for measuring throughput,
//! latency distributions, and memory footprints per AXIOM design doc §6.

pub mod memory;

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub kernel_name: String,
    pub dimension: String,
    pub throughput_gflops: f64,
    pub mean_latency_us: f64,
    pub p50_latency_us: f64,
    pub p99_latency_us: f64,
    pub rss_mb: f64,
}

impl BenchmarkReport {
    pub fn print_markdown_row(&self) {
        println!(
            "| {:<20} | {:<12} | {:>10.2} GFLOPs/s | {:>8.2} µs | {:>8.2} µs | {:>8.2} µs | {:>8.2} MB |",
            self.kernel_name,
            self.dimension,
            self.throughput_gflops,
            self.mean_latency_us,
            self.p50_latency_us,
            self.p99_latency_us,
            self.rss_mb
        );
    }
}
