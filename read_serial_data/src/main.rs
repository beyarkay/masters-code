use std::time::Duration;

fn main() {
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
        read_port_data(port);
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
fn read_port_data(mut port: Box<dyn serialport::SerialPort>) {
    let mut serial_buf: Vec<u8> = vec![0; 32];
    let mut s: String = "".to_string();
    let mut vals = vec![];
    let mut min = 0;
    let mut max = 800;
    loop {
        // If we've got a valid line of data
        if let Ok(num_bytes) = port.read(serial_buf.as_mut_slice()) {
            for idx in 0..num_bytes {
                let c = serial_buf[idx] as char;
                // check if we've reached the end of the line
                if c == '\n' {
                    println!("\n\nRaw values:           {:?}", vals);
                    let short_vals = vals.clone().into_iter().skip(3).collect::<Vec<i32>>();
                    if short_vals.len() == 15 {
                        min = if short_vals.len() != 0 { i32::min(min, *short_vals.iter().min().unwrap()) } else { min };
                        max = if short_vals.len() != 0 { i32::max(max, *short_vals.iter().max().unwrap()) } else { max };
                        println!("{}", values_per_finger(&short_vals, min, max));
                        println!("{}", values_per_dimension(&short_vals, min, max));
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
        }
    }
}

/// Return a sparkline like `▃▆▂▃▂▃▃` which can be used as a graph. 
///
/// The values in `data` are scaled between `low` and `high` and then mapped to 9 values which are
/// the sparks: ` ▁▂▂▄▅▆▇█` (note the inclusion of the space ` `).
fn spark(data: &Vec<i32>, low: i32, high: i32) -> String {
    assert!(low < high);
    let sparklines = vec![
        ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'
    ];
    let range = high - low;
    let step = range as f32 / sparklines.len() as f32;
    let mut sparkline = "".to_string();
    for datum in data {
        for (spark_idx, spark) in sparklines.iter().enumerate() {
            let lower = low as f32 + spark_idx as f32 * step;
            let upper = low as f32 + (spark_idx + 1) as f32 * step;
            if  lower <= *datum as f32 && *datum as f32  <= upper {
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

fn values_per_dimension(short_vals: &Vec<i32>, min: i32, max: i32) -> String {
    let mut s = format!("Values per dimension: ");
    let mut means = vec![0.0; 3];
    for (i, dim) in vec!["x", "y", "z"].iter().enumerate() {
        let mut dim_vec = vec![];
        for val in short_vals.iter().skip(i).step_by(3) {
            dim_vec.push(*val);
        }
        s.push_str(format!("{}: {} ", dim, spark(&dim_vec, min, max)).as_str());
        means[i] = dim_vec.iter().sum::<i32>() as f32 / 5.0;
        s.push_str(if means[i] > 250.0 { "inc " } else { "dec " } );
    }
    // s.push_str("\n");
    // for mean in means {
    //     s.push_str(format!("{} ", mean).as_str());
    // }
    return s;
}
