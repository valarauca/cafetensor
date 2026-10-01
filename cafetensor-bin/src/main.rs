fn main() {
    eprintln!("cafetensor: tier {}", cafetensor_lib::tier_name());
    if let Err(e) = cafetensor_lib::self_test() {
        eprintln!("cafetensor: self test failed: {e:?}");
        std::process::exit(1);
    }
}
