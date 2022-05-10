use mouse_rs::{Mouse, types::keys::Keys};
use std::time::{Duration, SystemTime};
const CONTROL_MOUSE: bool = true;

fn main() {
    let mouse = Mouse::new();
    let portname = "/dev/tty.usbmodem11101";
    let mut port = serialport::new(portname, 57_600)
        .timeout(Duration::from_millis(1))
        .open();
    if let Err(_) = port {
        println!("No serial found on `{}`", portname);
        // if the default port can't be found, then choose the last of the available ports
        let ports = serialport::available_ports().expect("No ports found!");
        let portname = ports.last().expect("No ports found").port_name.clone();
        println!("Reading from `{}` instead", portname.clone());
        port = serialport::new(portname, 57_600)
            .timeout(Duration::from_millis(1))
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
    let mut serial_buf: Vec<u8> = vec![0; 32];
    let mut s: String = "".to_string();
    let mut vals = vec![];
    let mut min = 0;
    let mut max = 800;
    loop {
        // sleep(Duration::from_millis(10));
        // If we've got a valid line of data
        if let Ok(num_bytes) = port.read(serial_buf.as_mut_slice()) {
            for idx in 0..num_bytes {
                let c = serial_buf[idx] as char;
                // check if we've reached the end of the line
                if c == '\n' {
                    println!("\n\nRaw values:           {:?}", vals);
                    let short_vals = vals.clone().into_iter().skip(2).collect::<Vec<i32>>();
                    if short_vals.len() == 15 {
                        min = if short_vals.len() != 0 {
                            i32::min(min, *short_vals.iter().min().unwrap())
                        } else {
                            min
                        };
                        max = if short_vals.len() != 0 {
                            i32::max(max, *short_vals.iter().max().unwrap())
                        } else {
                            max
                        };
                        println!("{}", values_per_finger(&short_vals, min, max));
                        println!("{}", values_per_dimension(&short_vals, min, max));
                        // write_to_file("", &short_vals);
                        if let Some(mouse) = mouse {
                            control_mouse(&short_vals, mouse);
                        }
                    }

                    vals = vec![];
                    s = "".to_string();
                } else if c != ',' {
                    // if we've just got another character
                    s.push(c);
                } else {
                    // We've reached the end of a number, and can parse it as one
                    vals.push(s.parse::<i32>().expect("Failed to parse string"));
                    s = "".to_string();
                }
            }
        } else {
            println!("No data found");
            break;
        }
    }
}

/// Return a sparkline like `▃▆▂▃▂▃▃` which can be used as a graph.
///
/// The values in `data` are scaled between `low` and `high` and then mapped to 9 values which are
/// the sparks: ` ▁▂▂▄▅▆▇█` (note the inclusion of the space ` `).
fn spark(data: &Vec<i32>, low: i32, high: i32) -> String {
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

fn values_per_finger(short_vals: &Vec<i32>, min: i32, max: i32) -> String {
    let mut s = format!("Values per finger:    ");
    for (i, chunk) in short_vals.chunks(3).enumerate() {
        let sparklines = spark(&chunk.to_vec(), min, max);
        s.push_str(format!("{}: {} ", i + 1, sparklines).as_str());
    }
    return s;
}

fn values_per_dimension(
    short_vals: &Vec<i32>,
    min: i32,
    max: i32,
) -> String {
    let mut s = format!("Values per dimension: ");
    let mut means = vec![0.0; 3];
    for (i, dim) in vec!["x", "y", "z"].into_iter().enumerate() {
        let mut dim_vec = vec![];
        for val in short_vals.iter().skip(i).step_by(3) {
            dim_vec.push(*val);
        }
        means[i] = dim_vec.iter().sum::<i32>() as f32 / 5.0;
        let accel_str;
        if means[i] > max as f32 * 0.60 {
            accel_str = "(incr)  ";
        } else if means[i] < max as f32 * 0.48 {
            accel_str = "(decr) ";
        } else {
            accel_str = "(steady)";
        }
        s.push_str(format!("{} {}: {} ", dim, accel_str, spark(&dim_vec, min, max)).as_str());
    }
    // s.push_str("\n");
    // for mean in means {
    //     s.push_str(format!("{} ", mean).as_str());
    // }
    return s;
}

fn control_mouse(
    short_vals: &Vec<i32>,
    mouse: &mut Mouse,
) {
    let pos = mouse.get_position().expect("Couldn't get mouse position");
    let mut delta = (0, 0);
    for (i, val) in short_vals.iter().enumerate() {
        if i / 3 == 0 && i % 3 == 2 { // Thumb z
            let x_min = 425;
            let x_max = 495;
            delta.0 = if *val < x_min { -1 } else { if *val > x_max { 1 } else { 0 } };
            delta.0 = if pos.x as i32 + delta.0 < 0 { 0 } else { delta.0 };
        }
        if i / 3 == 0 &&  i % 3 == 1 { // Thumb y
            let y_min = 465;
            let y_max = 525;
            delta.1 = if *val < y_min { 1 } else { if *val > y_max { -1 } else { 0 } };
            delta.1 = if pos.y as i32 + delta.1 < 0 { 0 } else { delta.1 };
        } 
        if i / 3 == 1 && i % 3 == 1 { // forefinger y
            let x_thresh = 500;
            if *val > x_thresh {
                //lclick_time = SystemTime::now();
                // println!("CLICK");
                mouse.click(&Keys::LEFT).ok();
            }
        }
        if i / 3 == 2 && i % 3 == 1 { // middle finger y
            let x_thresh = 500;
            if *val > x_thresh {
                // println!("CLICK");
                mouse.click(&Keys::RIGHT).ok();
            }
        }
    }
    mouse.move_to( 
        (pos.x as i32 + delta.0).try_into().unwrap(),
        (pos.y as i32 + delta.1).try_into().unwrap()
    ).ok();
}

fn _write_to_file(_data: &Vec<i32>, _filename: String) {
    todo!();
}
