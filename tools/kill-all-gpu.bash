for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do
    if ! ps -p $pid -o cmd= | grep -q '/usr/lib/xorg/Xorg'; then
        echo "Killing process PID: $pid"
        sudo kill -9 $pid
    else
        echo "Skipping Xorg process PID: $pid"
    fi
done

