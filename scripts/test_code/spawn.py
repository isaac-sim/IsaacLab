# use a urdf converter to conver a urdf to usd.


from isaaclab.sim.converters.urdf_converter import UrdfConverter, UrdfConverterCfg


if __name__ == "__main__":
    print("Converting urdf to usd...")
    urdf_converter_config = UrdfConverterCfg(
        asset_path="/home/arl/lmf2.urdf",
        usd_dir="/home/arl/lmf2",
        usd_file_name="lmf2.usd",
        force_usd_conversion=True,
        fix_base=False
    )
    print("Urdf converter config:", urdf_converter_config)
    urdf_converter = UrdfConverter(urdf_converter_config)
    print("[DONE] Converted urdf to usd.")
    print("Saved to:", urdf_converter.usd_path)
