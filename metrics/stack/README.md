# Metric

Infrastructura con docker compose para la recoleccion de metricas sobre los conteiners. Se utilizan: prometheus para la recoleccion y guardado de las metricas timeseries, cAdvisor como exportador de metricas de los containers de docker y grafana para la visualizacion de los datos.

## Run infrastucture

```bash
# start containers
docker-compose up -d

# stop containers
docker-compose down
```
