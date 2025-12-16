#!/bin/bash

# --- Configuration ---
SCRIPT_NAME="data_collection.py"
SPEC_NAME="data_collection.spec"
APP_NAME="data_collection.app"
BUNDLE_ID="com.fretlistener.datacollection"
USAGE_DESCRIPTION="This application needs access to your microphone to capture instrument audio for the dataset."

# Path Configuration (Assuming script is in fretListener/tools)
ARTIFACTS_DIR="../artifacts/tools"

# --- Step 1: Clean Up (Initial) ---
echo "1. Cleaning up previous build files..."
rm -rf build dist "$SPEC_NAME"

# --- Step 2: Generate the Specification File ---
echo "2. Generating spec file..."
pyinstaller --windowed \
            --exclude-module=torch --exclude-module=tensorflow \
            --collect-all sounddevice --collect-all soundfile --collect-all librosa \
            "$SCRIPT_NAME" --noconfirm

# --- Step 3: Modify the Spec File (Fixing indentation) ---
echo "3. Injecting Microphone Usage Description..."
python3 - <<EOF
import os

spec_file = "$SPEC_NAME"
with open(spec_file, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "bundle_identifier=None," in line:
        # Replace with properly indented lines
        new_lines.append(f"    bundle_identifier='$BUNDLE_ID',\n")
        new_lines.append(f"    info_plist={{'NSMicrophoneUsageDescription': '$USAGE_DESCRIPTION'}},\n")
    else:
        new_lines.append(line)

with open(spec_file, 'w') as f:
    f.writelines(new_lines)
EOF

# --- Step 4: Rebuild the Tool ---
echo "4. Rebuilding application..."
pyinstaller "$SPEC_NAME" --noconfirm

# --- Step 5: Artifact Management ---
echo "5. Managing artifacts..."

# Create the artifacts/tools directory if it doesn't exist
mkdir -p "$ARTIFACTS_DIR"

# Move the .app to the artifacts directory
# Note: PyInstaller creates the .app inside dist/
if [ -d "dist/$APP_NAME" ]; then
    echo "   -> Moving $APP_NAME to $ARTIFACTS_DIR"
    rm -rf "$ARTIFACTS_DIR/$APP_NAME" # Clean old version if exists
    mv "dist/$APP_NAME" "$ARTIFACTS_DIR/"
    
    # Create the ZIP file
    echo "   -> Creating ZIP archive..."
    # We cd into the directory so the zip doesn't contain the whole path tree
    (cd "$ARTIFACTS_DIR" && zip -r "${APP_NAME}.zip" "$APP_NAME" > /dev/null)
    echo "   -> Created: $ARTIFACTS_DIR/${APP_NAME}.zip"
else
    echo "❌ Error: $APP_NAME not found in dist/"
    exit 1
fi

# --- Step 6: Final Cleanup ---
echo "6. Final cleanup..."
rm -rf build dist "$SPEC_NAME"

echo "------------------------------------------------------"
echo "✅ BUILD & PACKAGING COMPLETE"
echo "Location: $ARTIFACTS_DIR/"
echo "Files: $APP_NAME, ${APP_NAME}.zip"
echo "------------------------------------------------------"