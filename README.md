# How to Set Up MonsterVision2 for Development
This document covers installing MV2 on a Raspberry Pi development machine. See (???)) for deployment instructions.

It is recommeneded (but not required) that you use an SSD rather than an SD card on your Pi.  If you do, you may need to enable your Pi to boot from the SSD.  This only needs to be done once.  [Follow these instructions.](https://peyanski.com/how-to-boot-raspberry-pi-4-from-ssd/#:~:text=To%20boot%20Raspberry%20Pi%204%20from%20SSD%20you,USB%20to%20boot%20raspberry%20Pi%204%20from%20SSD.)

Once you've gotten your Pi up and running, follow this procedure:

## Start a Terminal session.
Within the session:

Make sure `pip` is up-to-date:
```shell
pip install --upgrade pip
``` 
Clone the MonsterVision2 repo:
```shell
git clone https://github.com/Lakemonsters2635/MonsterVision2.git
```
Change to the MonsterVision2 directory:
```shell
cd MonsterVision2
```

For development, it is best to use a Python virtual environment to keep from descending into "version hell."  Create the virtual environment and activate it.
```shell
virtualenv env
. env/bin/activate
```

Install all of the requirements for running:
```shell
pip install -r requirements.txt
```
Note that on a fresh system, the installation of OpenCV (opencv-contrib-python==4.5.5.62) may take several hours to build on an SSD-based system - even longer if you are using an SD card.

Finally, you'll need to copy 2 files into the `/boot` directory.  You'll need root permission to do this.  The first file is `frc.json` and contains:
```json
{
    "cameras": [],
    "ntmode": "client",
    "switched cameras": [],
    "team": 2635,
    "hasDisplay": 1
}
```
|Entry|Values||
|---|---|---|
|`ntmode`|**client**|Network tables server hosted remotely|
||**server**|Network tables server hosted locally|
|`team`|Team number||
|`hasDisplay`|**0**|Host is headless|
||**1**|Host has attached display - depth and annotation windows will be displayed|

Copy this file:
```shell
sudo cp frc.json /boot
```
The second needed in /boot is `nn.json`.  This file determines which detection network is to be run.  Copy one of the `.json` files from `./models` directory.  For example, to use the 2022 Cargo YOLOv6 Tiny network:
```shell
sudo cp models/nn-2022cargoyolo6t.json /boot/nn.json
```
Run MonsterVision2 via:
```shell
python MonsterVision2.py
```
## Development Environment
Visual Studio Code is the preferred development environment for consistency with our general FRC code development.  It is officially distributed via the Raspberry Pi OS APT repository in both 32- and 64-bit versions:  Install via:
```shell
sudo apt update
sudo apt install code
```
It can be launched with from a terminal via `code` or via the GUI under the **Programming** menu.
### Install ***GitHub Pull Requests and Issues*** Externsion
To enable GitHub integration, you need to install the above-named extension.  After installation, you'll need to log into your GitHub account.  **TODO** Instructions for doing this.
### Debugging Using VS Code
- After launching VS Code, select **File** | **Open Folder...** and select the `MonsterVision2` directory.
- Select **View** | **Command Palette...**.
- Choose `Python: Select Interpreter`.
- From the list of interpreters, choose the one in your virtual environment.  It will look something like this: `Python n.n.n ('env':venv) ./env/bin/python`
- From the left pane, select `MonsterVision2.py` to open it.
Either hit `F5` or select **Run**|**Start Debugging** to run MonsterVision2 under control of the debugger.
# How to Install MonsterVision2 for FRC Competition
Worcester Polytechnic Institute (WPI), the creators and maintainers of the WPI Library we use for FRC robot development have create WPILibPi, a derivative of the Raspberry Pi OS, Raspbian. This Raspbian-based Raspberry Pi image includes C++, Java, and Python libraries required for vision coprocessor development for FRC (e.g. opencv, cscore, ntcore, robotpy-cscore, pynetworktables, Java 11, etc). WPILibPi comes in two variants, the "base" image for vision coprocessors, and an image designed for use with Pololu Romi 32U4 based robots.  Please follow these instructions to [install WPILibPi on an SD card.](https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/installing-the-image-to-your-microsd-card.html).  Make sure you have the latest release.  Follow all instructions on the web page up to and including last step where you log into a remote terminal.  Use the SSH command from a command prompt:
```shell
ssh wpilibpi.local -l pi
```
**IMPORTANT** Do not skip this step. You must set the correct date.  Otherwise, many of the following commands will fail:
```shell
sudo date MMDDhhmmYYYY
```
Where MM is the month, DD is the day, hh is the hours (24-hour time), mm is the minutes and YYYY is the year.  For example:
```shell
sudo date 010323302023
```
The system should then print out the date you just set.

Once logged into the Raspberry Pi, there are a few things that need to be done just once.  We need to install some additional software.  We also need to upgrade the WPI-supplied version of numpy to a specific version:
```shell
sudo apt update
sudo apt install git
sudo apt remove python3-numpy
python3 -m pip install "numpy>=1.20,<=1.22"
sudo apt-get install python3-h5py
sudo apt-get install libatlas-base-dev
```
We need to enable access to the USB-connected camera:
```shell
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Next, download MonsterVision2:
```shell
git clone https://github.com/LakeMonsters2635/MonsterVision2.git
```
Now we need to install the required Python modules:
```shell
cd MonsterVision2
python3 -m pip install -r requirementsWPI.txt
```
Now we need to configure the Network Tables.  Point your browser to http://wpilibpi.local.  This loads the web interface.  Note that the first time you access this page, the SD card will be marked Writable (the buttons at the top of the web page).  On subsequent accesses, you need to click that button if you want to write to the Pi.

Click on `Network Settings` in the left pane.  Make sure the `Client` switch is turned on and the `Team Number` is set to your team number.

Return now to your SSH session so we can configure the correct detection network. Copy one of the `.json` files from `./models` directory.  For example, to use the 2022 Cargo YOLOv6 Tiny network:
```shell
sudo cp models/nn-2022cargoyolo6t.json /boot/nn.json
```
We also need to replace the stock WPILibPi camera module with MonsterVision2 and make MonsterVision2 executable
```shell
cp runCamera ..
chmod +x MonsterVision2.py
```
Finally, reboot the Pi:
```shell
sudo reboot +0
```
You're SSH session will be disconnected from the Pi.  If you want to reconnect, wait about 30 seconds before trying.  However, you should be able to do everything from here on from the browser interface.

Once again, point your browser to http://wpilib.local.  Click on the **Vision Status** button in the left pane.  You can use the 4 buttons (**Up**, **Down**, **Terminate**, and **Kill**) to control MonsterVision2.  It can also be useful to enable Console Output.  That way, if you need to restart MonsterVision2, you can see its printed output.

You can view a reduced-framerate version of the annotated RGB camera stream by browsing http://wpilib.local:1181.  This same stream is also available via the Shuffleboard.