import subprocess
import time
import sys
import os

framework = sys.argv[1] if len(sys.argv) > 1 else "default_framework"
model = sys.argv[2] if len(sys.argv) > 2 else "default_model"
operation = sys.argv[3] if len(sys.argv) > 3 else "default_operation"

cmd_query = [
    'nvidia-smi',
    '--id=GPU-0d58720e-34f6-3fd5-510d-e6d5249693f4',
    '--query-gpu=index,uuid,name,memory.total,memory.reserved,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw.instant,power.limit',
    '--format=csv'
]

cmd_tail = ['tail', '-n', '+2']

headers_mapper = {
    'memory.total [MiB]': 'memory_total',
    'memory.reserved [MiB]': 'memory_reserved',
    'memory.used [MiB]': 'memory_used',
    'memory.free [MiB]': 'memory_free',
    'utilization.gpu [%]': 'utilization_gpu',
    'utilization.memory [%]': 'utilization_memory',
    'temperature.gpu': 'temperature_gpu',
    'temperature.memory': 'temperature_memory',
    'power.draw.instant [W]': 'power_draw_instant',
    'power.limit [W]': 'power_limit'
}

def query_gpu_status(first_call=False):
    # Ejecutar el comando y capturar la salida
    try:
        if first_call:
            result = subprocess.run(cmd_query, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            process_nvidia_smi = subprocess.Popen(cmd_query, stdout=subprocess.PIPE)
            process_tail = subprocess.Popen(cmd_tail, stdin=process_nvidia_smi.stdout, stdout=subprocess.PIPE)
            output, _ = process_tail.communicate()
            return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        # Si ocurre un error al ejecutar nvidia-smi, captura el error y retorna un mensaje
        return f"Error executing nvidia-smi: {e}"

def main():
    # Abrir el archivo para guardar la salida
    filename = "gpu_status_log.csv"
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0


    with open(filename, "a") as f:
        if not file_exists:
            output = query_gpu_status(True)
            headers = output.splitlines()[0]  # La primera línea contiene los encabezados
            headers = headers.split(', ')
            headers_mapped = [headers_mapper.get(header, header) for header in headers]
            headers_mapped = ','.join(headers_mapped)
            f.write("framework,model,operation,timestamp," + headers_mapped + "\n")
        while True:
            # Obtener la salida de nvidia-smi
            timestamp = round(time.time() * 1000)
            output = query_gpu_status(False)
            output = output.replace(' MiB', '').replace(' %', '').replace(' W', '').replace(', ', ',')
            f.write(f'{framework},{model},{operation},{timestamp},' + output)
            f.flush()
            time.sleep(1)

if __name__ == "__main__":
    main()
