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

fn read_port_data(mut port: Box<dyn serialport::SerialPort>) {
    let mut serial_buf: Vec<u8> = vec![0; 32];
    let mut s: String = "".to_string();
    let mut vals = vec![];
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    loop {
        // If we've got a valid line of data
        if let Ok(num_bytes) = port.read(serial_buf.as_mut_slice()) {
            for idx in 0..num_bytes {
                let c = serial_buf[idx] as char;
                // check if we've reached the end of the line
                if c == '\n' {
                    println!("\n\nRaw values: {:?}", vals);
                    let short_vals = vals.clone().into_iter().skip(3).collect::<Vec<i32>>();
                    // println!("min: {}, max: {}", min, max);
                    min = i32::min(min, *short_vals.iter().min().expect("Vector was empty"));
                    max = i32::max(max, *short_vals.iter().max().expect("Vector was empty"));
                    print!("Values per finger: ");
                    for (i, chunk) in short_vals.chunks(3).enumerate() {
                        let sparklines = spark(chunk.to_vec(), min, max);
                        print!("{}: {} ", i + 1, sparklines);
                    }
                    print!("\nValues per dimension: ");
                    for (i, dim) in vec!["x", "y", "z"].iter().enumerate() {
                        let mut dim_vec = vec![];
                        for val in short_vals.iter().skip(i).step_by(3) {
                            dim_vec.push(*val);
                        }
                        print!("{}: {} ", dim, spark(dim_vec, min, max));
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

fn spark(vec: Vec<i32>, low: i32, high: i32) -> String {
    assert!(low < high);
    let sparklines = vec![
        ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'
    ];
    let range = high - low;
    let step = range as f64 / sparklines.len() as f64;
    let mut sparkline = "".to_string();
    for i in vec {
        for (spark_idx, spark) in sparklines.iter().enumerate() {
            let lower = low as f64 + spark_idx as f64 * step;
            let upper = low as f64 + (spark_idx + 1) as f64 * step;
            if  lower <= i as f64 && i as f64  <= upper {
                sparkline.push(*spark);
            }
        }
    }
    return sparkline;
}
