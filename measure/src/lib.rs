use std::fs::File;
use std::io::{self, Write};

pub const MEASURE_START: usize = 0;
pub const MEASURE_CSER: usize = 1;
pub const MEASURE_CSEND: usize = 2;
pub const MEASURE_SRECV: usize = 3;
pub const MEASURE_SDSER: usize = 4;
pub const MEASURE_RAW: usize = 5;
pub const MEASURE_SSER: usize = 6;
pub const MEASURE_SSEND: usize = 7;
pub const MEASURE_CRECV: usize = 8;
pub const MEASURE_CDSER: usize = 9;
pub const MEASURE_TOTAL: usize = 10;
const MEASURE_MAX_NUM: usize = 11;

const CLOCK_FREQUENCY: f64 = 2.2;
const ITER_NUM: usize = 10010;

pub struct Timer {
    start_time: [[u64; MEASURE_MAX_NUM]; ITER_NUM],
    stop_time: [[u64; MEASURE_MAX_NUM]; ITER_NUM],
    cnt: usize,
    output_file: String,
}

#[inline]
pub fn rdtscp() -> u64 {
    let mut aux = 0;
    unsafe { core::arch::x86_64::__rdtscp(&raw mut aux) }
}

#[inline]
pub fn clock2ns(clock: u64) -> f64 {
    clock as f64 / CLOCK_FREQUENCY
}

impl Timer {
    pub fn new(file_name: String) -> Self {
        Timer {
            start_time: [[0; MEASURE_MAX_NUM]; ITER_NUM],
            stop_time: [[0; MEASURE_MAX_NUM]; ITER_NUM],
            cnt: 0,
            output_file: file_name,
        }
    }

    #[inline]
    pub fn set(&mut self, id: usize) {
        self.start_time[self.cnt][id] = rdtscp();
    }

    #[inline]
    pub fn stop(&mut self, ty: usize) {
        self.stop_time[self.cnt][ty] = rdtscp();
    }

    #[inline]
    pub fn plus_cnt(&mut self) {
        self.cnt += 1;
        if self.cnt == ITER_NUM {
            let _ = self.write();
        }
    }

    pub fn write(&self) -> io::Result<()> {
        let mut file = File::create(self.output_file.clone())?;
        for i in 0..ITER_NUM {
            let mut row_result = Vec::new();
            for j in 0..MEASURE_MAX_NUM {
                row_result.push(self.start_time[i][j].to_string());
            }
            writeln!(file, "{}", &row_result.join(", "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_test() {
        let mut timer = Timer::new();

        timer.start(MEASURE_TOTAL);
        let mut _sum: u64 = 0;
        for i in 0..10000 {
            _sum += i;
        }
        timer.stop(MEASURE_TOTAL);
        assert!(timer.get_time(MEASURE_TOTAL) > 0.0);
    }
}
