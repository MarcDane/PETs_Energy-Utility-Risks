import pyRAPL
import time

NR_OF_MEASUREMENTS = 10

output_file = 'energy/idle.csv'

def idle():
	pyRAPL.setup()
	csv_output = pyRAPL.outputs.CSVOutput(output_file)

	with pyRAPL.Measurement('idle', output=csv_output):
		time.sleep(10)

	csv_output.save()

for i in range(NR_OF_MEASUREMENTS):
	print(f'Idle Measurement {i} ...')
	idle()
	time.sleep(2)
