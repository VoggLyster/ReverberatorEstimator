curl -s https://api.github.com/repos/VoggLyster/Reverberator/releases/latest \
| grep "browser_download_url.*zip" \
| cut -d : -f 2,3 \
| tr -d \" \
| wget -qi -
unzip Reverberator.zip
rm -rf Reverberator.vst3
mv Builds/LinuxMakefile/build/Reverberator.vst3 Reverberator.vst3
rm -rf Builds
rm -rf Reverberator.zip