# Emotion Based IoT
 Control a Raspberry Pi by your real time facial emotion.  
 This is a project made by me, with references to public code and some electrical stuffs.  
 This program read video stram then use facial detection, crop face, then feed to vgg19 for emotion classification, then transmit the emotion through the Internet to the RaspberryPi using NAT traverse tech, the daemon service on RaspberryPi receive the emotion distribution, then output GPIO signal to control electric relays to control different color of LED to light the room.
