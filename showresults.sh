while true;
do
	grep "Generation best" log.txt | grep -o -P '.{0,6}%.{0}' | tr -d '%' | gnuplot -p -e 'set terminal dumb; plot "/dev/stdin" using 0:1 with lines'
	cat `ls -tr *.png | tail -n 1` | display -geometry -100+150 &
	DISPLAY_PID=$!
	sleep 60
	kill "$DISPLAY_PID"
done
