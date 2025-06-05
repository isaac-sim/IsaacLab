import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf
import os # For path operations

def set_joint_drive_properties(stage, articulation_prim_path, stiffness, damping):
    """
    Iterates through the joints of an articulation and sets the stiffness and damping
    for the angular drive.

    Args:
        stage (Usd.Stage): The USD stage.
        articulation_prim_path (str): The path to the root prim of the robot articulation.
        stiffness (float): The desired stiffness value.
        damping (float): The desired damping value.
    """
    root_prim = stage.GetPrimAtPath(articulation_prim_path)

    if not root_prim.IsValid():
        print(f"Error: Articulation Prim not found at path for setting drives: {articulation_prim_path}")
        return

    print(f"\nStarting to set joint drive properties for: {articulation_prim_path}")
    joints_configured_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            try:
                angular_drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not angular_drive:
                    print(f"  Info: No angular drive found for joint {prim.GetPath()}. Applying DriveAPI.")
                    angular_drive = UsdPhysics.DriveAPI.Apply(prim, "angular")

                stiffness_attr = angular_drive.GetStiffnessAttr()
                damping_attr = angular_drive.GetDampingAttr()

                if stiffness_attr and damping_attr:
                    stiffness_attr.Set(stiffness)
                    damping_attr.Set(damping)
                    print(f"  Set stiffness ({stiffness}) and damping ({damping}) for joint: {prim.GetPath()}")
                    joints_configured_count += 1
                else:
                    print(f"  Error: Could not get stiffness/damping attributes for joint {prim.GetPath()}")
            except Exception as e:
                print(f"  Could not set properties for joint {prim.GetPath()}: {e}")
    print(f"Finished setting joint drive properties. Configured {joints_configured_count} joints.")


def add_contact_reporter_by_name_matching(stage, robot_root_path, colliders_folder_path, report_threshold=0.0):
    """
    Finds prims in a 'colliders' folder, then finds prims with matching names
    within the robot hierarchy and applies a contact reporter to them.
    """
    collider_names = set()
    colliders_prim = stage.GetPrimAtPath(colliders_folder_path)
    
    if not colliders_prim.IsValid():
        print(f"Error: Colliders folder not found at path: '{colliders_folder_path}'")
        return

    for prim in colliders_prim.GetChildren():
        collider_names.add(prim.GetName())
    
    if not collider_names:
        print(f"Warning: No prims were found inside the colliders folder at '{colliders_folder_path}'.")
        return

    print(f"\nFound {len(collider_names)} collider names to match for contact reporters: {collider_names}")

    robot_root_prim = stage.GetPrimAtPath(robot_root_path)
    if not robot_root_prim.IsValid():
        print(f"Error: Robot root prim not found at path for contact reporters: '{robot_root_path}'")
        return

    print(f"Processing robot at '{robot_root_path}' for contact reporters...")
    links_modified_count = 0
    
    for link_prim in Usd.PrimRange(robot_root_prim):
        link_name = link_prim.GetName()
        if link_name in collider_names:
            print(f"  Found matching link for contact reporter: '{link_prim.GetPath()}' (name: '{link_name}')")
            try:
                contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                if contact_report_api:
                    threshold_attr = contact_report_api.CreateThresholdAttr()
                    threshold_attr.Set(report_threshold)
                    print(f"    Added/Updated PhysxContactReportAPI with threshold {report_threshold}")
                    links_modified_count += 1
                else:
                    print(f"    Failed to apply PhysxContactReportAPI.")
            except Exception as e:
                print(f"    Error applying API to link '{link_prim.GetPath()}': {e}")
    print(f"Finished adding contact reporters. Modified {links_modified_count} links.")

def save_stage_as_usd(stage, file_name_prefix, output_directory="C:/temp/isaac_saves"):
    """
    Saves the current stage to a USD file.

    Args:
        stage (Usd.Stage): The USD stage to save.
        file_name_prefix (str): A prefix for the output file name (e.g., "robot_configured").
        output_directory (str): The directory where the file will be saved.
                                Defaults to "C:/temp/isaac_saves".
                                This directory will be created if it doesn't exist.
    """
    if not stage:
        print("Error: Invalid stage provided for saving.")
        return
    if not file_name_prefix:
        print("Error: file_name_prefix cannot be empty for saving.")
        return

    try:
        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Info: Created output directory: {output_directory}")

        # Construct the output file path
        # Using .usda for human-readable, .usdc for binary (smaller, faster load)
        file_name = f"{file_name_prefix}.usd" # Or .usda
        output_file_path = os.path.join(output_directory, file_name)

        print(f"\nAttempting to save stage to: {output_file_path}")
        # Export the stage
        # For anonymous layers, you might need to export the root layer: stage.GetRootLayer().Export(output_file_path)
        # For stages opened from a file, stage.Export() should work.
        # Using GetRootLayer().Export() is generally safer for scripts that modify the stage in memory.
        success = stage.GetRootLayer().Export(output_file_path)
        
        if success:
            print(f"Successfully saved stage to: {output_file_path}")
        else:
            print(f"Error: Failed to save stage to: {output_file_path}. Check console for USD errors.")
            # Additional USD error checking can be added here if needed
            # error_observer = Usd.ErrorObserver()
            # if error_observer.GetErrors():
            #     for err in error_observer.GetErrors():
            #         print(f"  USD Error: {err.GetErrorCodeAsString()}: {err.GetCommentary()}")


    except Exception as e:
        print(f"Error during save operation: {e}")


if __name__ == "__main__":
    stage = omni.usd.get_context().get_stage()

    if not stage:
        print("Error: Could not get the current USD stage. Make sure a stage is open.")
    else:
        
        
        # --- Configuration ---
        # IMPORTANT: Set the path to your robot's root prim here
        robot_prim_path = "/h1_2"  # <--- CHANGE THIS TO YOUR ROBOT'S ACTUAL PATH

        #Set your stiffness and damping values
        preset_stiffness = 0.0
        preset_damping = 0.0  

        colliders_folder_path = "/colliders" 

        #Set your contact reporter value
        contact_report_threshold = 0.0     

        # --- Save Configuration ---
        # IMPORTANT: Set your desired file name prefix and output directory
        #Set the file name to the same as the URDF 
        output_file_name_prefix = "/h1_2" 
        # Example: "C:/Users/YourUser/Documents/IsaacSimExports"
        # Make sure the path is valid for your OS.
        # Using a generic temp directory for this example, but customize it.
        # Ensure this directory exists or the script has permissions to create it.
        output_save_directory = "C:/Users/khammoud/Git/unitree/unitree_rl_gym/resources/robots/h1_2/h1_2" # Windows example
        

        if not robot_prim_path: 
            print("ERROR: robot_prim_path is not set. Please edit the script.")
        else:
            print(f"--- Starting Robot Configuration Script for: {robot_prim_path} ---")
            
            set_joint_drive_properties(stage, robot_prim_path, preset_stiffness, preset_damping)
            add_contact_reporter_by_name_matching(stage, robot_prim_path, colliders_folder_path, contact_report_threshold)
            
            # --- Save the modified stage ---
            save_stage_as_usd(stage, output_file_name_prefix, output_save_directory)

            print(f"\n--- Robot Configuration Script Finished for: {robot_prim_path} ---")
