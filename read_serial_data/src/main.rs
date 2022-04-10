use std::time::Duration;
use serialport;

fn main() {
    let mut port = serialport::new("/dev/tty.usbmodem11101", 57_600)
        .timeout(Duration::from_millis(1))
        .open().expect("Failed to open port");

    let mut serial_buf: Vec<u8> = vec![0; 64];
    loop {
        if let Ok(num_bytes) = port.read(serial_buf.as_mut_slice()) {
            for idx in 0..num_bytes {
                print!("{}", serial_buf[idx] as char);
            }
        } else {
            println!("No data found");
        }
    }
}
