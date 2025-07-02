for temperature in 0.1 0.3 0.5 0.8 1.0; do
    for timesteps in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32; do
        echo "temperature=${temperature}, timesteps=${timesteps}"
        python inference.py --temperature ${temperature} --timesteps ${timesteps}
    done
done