use chrono::prelude::{DateTime, Local};
use mouse_rs::{types::keys::Keys, Mouse};
use std::io::Write;
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    fs::{self, File},
    path::Path,
    thread,
    time::{Duration, SystemTime},
};
const CONTROL_MOUSE: bool = false;

fn main() {
    let mouse = Mouse::new();
    let portname = "/dev/tty.usbmodem11101";
    let baudrate = 19_200;
    let mut port = serialport::new(portname, baudrate)
        .timeout(Duration::from_millis(100))
        .open();
    if let Err(_) = port {
        println!("No serial found on `{}`", portname);
        // if the default port can't be found, then choose the last of the available ports
        let ports = serialport::available_ports().expect("No ports found!");
        let portname = ports.last().expect("No ports found").port_name.clone();
        println!("Reading from `{}` instead", portname.clone());
        port = serialport::new(portname, baudrate)
            .timeout(Duration::from_millis(50))
            .open();
    }

    // If the serial port exists
    if let Ok(port) = port {
        if CONTROL_MOUSE {
            read_port_data(port, &mut Some(mouse));
        } else {
            read_port_data(port, &mut None);
        }
    } else {
        // If we can't find the serial port, list the ones we can find
        let ports = serialport::available_ports().expect("No ports found!");
        println!(
            "Failed to read port `{}`, but found the following serial ports:",
            portname
        );
        for p in ports {
            println!("- {}", p.port_name);
        }
    }
}

/// Given a port as returned by `serialport::new(...).open().unwrap()`, read the incoming sensor
/// data from that port and plot it as a series of sparklines to `stdout`
fn read_port_data(mut port: Box<dyn serialport::SerialPort>, mouse: &mut Option<Mouse>) {
    let mut serial_buf: Vec<u8> = vec![0; 128];
    let mut val: String = "".to_string();
    let mut vals = vec![];
    let mut min = 0;
    let mut max = 800;
    let mut lclick_time = SystemTime::now();
    let mut rclick_time = SystemTime::now();
    let mut data: Vec<Vec<u16>> = vec![];
    let mut prev_millis = 0;
    let mut start = Local::now();
    let pbar = ProgressBar::new(1000);
    pbar.set_style(ProgressStyle::default_bar()
                  .template("{bar:40.cyan/black} {pos:>7}/{len:7} {msg}")
                  .progress_chars("=>-"));
    loop {
        // If we've got a valid line of data
        let reading = port.read(serial_buf.as_mut_slice());
        if let Ok(num_bytes) = reading {
            for idx in 0..num_bytes {
                let c = serial_buf[idx] as char;
                // check if we've reached the end of the line
                if c == '\n' {
                    if data.len() > 0 && data[0][0] == 0 {
                        println!("Raw values:           {:?}", vals);
                    }
                    // If we've received a full packet of information
                    if vals.len() != 32 {
                        vals = vec![];
                        val = "".to_string();
                        continue;
                    }

                    if prev_millis > vals[1] {
                        // TODO add checks to see if the gloves are in the resting position
                        // TODO This can be sped up if we don't try to parse the Serial input as
                        // numbers only to convert it back to strings for writing to file
                        //
                        if data.len() > 0 && data[0][0] > 0 {
                            pbar.set_position(1000);
                        }
                        if data[0][0] > 0 && data.len() > 35 {
                            let gesture_idx = data[0][0];
                            let dir = format!("../gesture_data/train/gesture{gesture_idx:0>4}/");
                            let paths: Vec<_> = fs::read_dir(dir.clone()).unwrap().collect();
                            write_to_file(data, start);
                            pbar.set_message(format!("Wrote observation; {} files in {}", paths.len(), dir));
                        }
                        start = Local::now();
                        data = vec![];
                    }
                    data.push(vals.clone());
                    prev_millis = vals[1];
                    if data.len() > 0 && data[0][0] > 0 {
                        pbar.set_position(prev_millis.into());
                    }
                    let short_vals = vals.clone().into_iter().skip(2).collect::<Vec<u16>>();
                    min = if short_vals.len() != 0 {
                        u16::min(min, *short_vals.iter().min().unwrap())
                    } else {
                        min
                    };
                    max = if short_vals.len() != 0 {
                        u16::max(max, *short_vals.iter().max().unwrap())
                    } else {
                        max
                    };
                    if vals[0] == 0 {
                        println!("{}", values_per_finger(&short_vals, min, max));
                        println!("{}", values_per_dimension(&short_vals, min, max));
                        println!("\n\n\n\n");
                    }

                    if let Some(mouse) = mouse {
                        control_mouse(&short_vals, mouse, &mut lclick_time, &mut rclick_time);
                    }

                    // reset the values and s string
                    vals = vec![];
                    val = "".to_string();
                } else if c != ',' {
                    // if we've just got another character
                    val.push(c);
                } else {
                    // We've reached the end of a number, and can parse it as one
                    vals.push(val.parse::<u16>().unwrap_or(0));
                    val = "".to_string();
                }
            }
        } else {
            println!("No data found: {:?}", reading.err().unwrap());
            break;
        }
    }
}

/// Return a sparkline like `▃▆▂▃▂▃▃` which can be used as a graph.
///
/// The values in `data` are scaled between `low` and `high` and then mapped to 9 values which are
/// the sparks: ` ▁▂▂▄▅▆▇█` (note the inclusion of the space ` `).
fn spark(data: &Vec<u16>, low: u16, high: u16) -> String {
    assert!(low < high);
    let sparklines = vec![' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let range = high - low;
    let step = range as f32 / sparklines.len() as f32;
    let mut sparkline = "".to_string();
    for datum in data {
        for (spark_idx, spark) in sparklines.iter().enumerate() {
            let lower = low as f32 + spark_idx as f32 * step;
            let upper = low as f32 + (spark_idx + 1) as f32 * step;
            if lower <= *datum as f32 && *datum as f32 <= upper {
                sparkline.push(*spark);
            }
        }
    }
    return sparkline;
}

fn values_per_finger(short_vals: &Vec<u16>, min: u16, max: u16) -> String {
    let mut s = format!("Values per finger:    ");
    for (i, chunk) in short_vals.chunks(3).enumerate() {
        let sparklines = spark(&chunk.to_vec(), min, max);
        s.push_str(format!("{}: {} ", i + 1, sparklines).as_str());
    }
    return s;
}

fn values_per_dimension(short_vals: &Vec<u16>, min: u16, max: u16) -> String {
    let mut s = format!("Values per dimension: ");
    let mut _means = vec![0.0; 3];
    for (i, dim) in vec!["x", "y", "z"].into_iter().enumerate() {
        let mut dim_vec = vec![];
        for val in short_vals.iter().skip(i).step_by(3) {
            dim_vec.push(*val);
        }
        // _means[i] = dim_vec.iter().sum::<u16>() as f32 / 5.0;
        let accel_str = "";
        // if _means[i] > max as f32 * 0.60 {
        //     accel_str = "(incr)  ";
        // } else if _means[i] < max as f32 * 0.48 {
        //     accel_str = "(decr) ";
        // } else {
        //     accel_str = "(steady)";
        // }
        s.push_str(format!("{} {}: {} ", dim, accel_str, spark(&dim_vec, min, max)).as_str());
    }
    // s.push_str("\n");
    // for mean in _means {
    //     s.push_str(format!("{} ", mean).as_str());
    // }
    return s;
}

fn control_mouse(
    short_vals: &Vec<u16>,
    mouse: &mut Mouse,
    lclick_time: &mut SystemTime,
    rclick_time: &mut SystemTime,
) {
    let pos = mouse.get_position().expect("Couldn't get mouse position");
    let mut delta = (0, 0);
    for (i, val) in short_vals.iter().enumerate() {
        if i / 3 == 0 && i % 3 == 2 {
            // Thumb z
            let x_min = 425;
            let x_max = 495;
            delta.0 = if *val < x_min {
                -1
            } else {
                if *val > x_max {
                    1
                } else {
                    0
                }
            };
            delta.0 = if pos.x as i32 + delta.0 < 0 {
                0
            } else {
                delta.0
            };
        }
        if i / 3 == 0 && i % 3 == 1 {
            // Thumb y
            let y_min = 465;
            let y_max = 525;
            delta.1 = if *val < y_min {
                1
            } else {
                if *val > y_max {
                    -1
                } else {
                    0
                }
            };
            delta.1 = if pos.y as i32 + delta.1 < 0 {
                0
            } else {
                delta.1
            };
        }
        if i / 3 == 1 && i % 3 == 1 {
            // forefinger y
            let x_thresh = 500;
            if *val > x_thresh
                && SystemTime::now().duration_since(*lclick_time).unwrap()
                    > Duration::from_millis(100)
            {
                *lclick_time = SystemTime::now();
                // println!("CLICK");
                mouse.click(&Keys::LEFT).ok();
            }
        }
        if i / 3 == 2 && i % 3 == 1 {
            // middle finger y
            let x_thresh = 500;
            if *val > x_thresh
                && SystemTime::now().duration_since(*rclick_time).unwrap()
                    > Duration::from_millis(100)
            {
                *rclick_time = SystemTime::now();
                // println!("CLICK");
                mouse.click(&Keys::RIGHT).ok();
            }
        }
    }
    mouse
        .move_to(
            (pos.x as i32 + delta.0).try_into().unwrap(),
            (pos.y as i32 + delta.1).try_into().unwrap(),
        )
        .ok();
}

fn write_to_file(data: Vec<Vec<u16>>, start: DateTime<Local>) {
    thread::spawn(move || {
        let gesture_idx = data[0][0];
        let measurements = data
            .into_iter()
            .map(|line| {
                // Convert each line into a csv string
                line.into_iter()
                    .skip(1) // Skip the gesture index
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<String>>();
        let dir = format!("../gesture_data/train/gesture{gesture_idx:0>4}");
        let filename = format!("{dir}/{}.txt", start.to_rfc3339());
        if !Path::new(&dir).is_dir() {
            fs::create_dir(dir).unwrap();
        }
        let file = File::create(filename.clone());
        match file {
            Ok(mut f) => {
                write!(&mut f, "{}", measurements.join("\n")).unwrap();
            }
            Err(e) => {
                println!("Failed to write to {}: {}", filename, e);
            }
        }
    });
}
