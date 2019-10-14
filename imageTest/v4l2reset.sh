contrast=57
saturation=72
hue=0
wbta=1
gamma=300
plf=0 #pass
wba=2800
sharpness=92
bc=1
eAuto=1
eAbu=180
eap=1

v4l2-ctl -d /dev/video1 -c contrast=$contrast -c saturation=$saturation -c hue=$hue -c white_balance_temperature_auto=$wbta -c gamma=$gamma -c white_balance_temperature=$wba -c sharpness=$sharpness -c backlight_compensation=$bc -c exposure_auto=$eAuto -c exposure_absolute=$eAbu -c exposure_auto_priority=$eap
