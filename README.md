# Construcción de la base de datos

Para la construcción de la base de datos preliminar se tienen en cuenta, hasta ahora, 4 fuentes de datos pincipales.

- **Casos confirmados de Dengue:** Casos confirmados de Dengue y Dengue Grave obtenidos por el registro en SIVIGILA y consolidado que dispone el Ministerio de Salud y Protección Social a través de los cubos de información del SISPRO. Se obtienen datos historicos con periodicidad mensual de 2007 a 2024, la agregación es a nivel municipal.
- **Coordenadas:** Para la obtención de coordenadas por municipio se utiliza el recurso del DANE con la codificación DIVIPOLA que se obtiene en https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/datos-geoestadisticos/?cod=112
- **Variables Climaticas:** Obtenidas de la base de datos historica mensual de WorldClim, quienes funcionan como un repositorio que consolida los reportes del clima de estaciones en todo el mundo, al generar datos espaciales del clima utilizan técnicas de interpolación para generar superficies continuas (splines de placa delgada) apoyándose en datos de elevación y distancia a la costa para estimar el clima en zonas sin estaciones.

    - **Splines de Placa Delgada (Thin Plate Splines):** Utilizan un software llamado ANUSPLIN. Este algoritmo ajusta superficies suaves a los datos de las estaciones. Imagina estirar una lámina de goma flexible que debe tocar la "cabeza" de cada estación meteorológica.
    - **Uso de Covariables (La Clave):** No interpolan solo basándose en la distancia entre estaciones. Usan variables auxiliares que se conocen con mucha precisión satelital:
    - **Elevación (DEM):** Es el factor más importante. Como la temperatura disminuye con la altura, WorldClim usa datos de elevación (como SRTM) para corregir la interpolación. Esto permite que el modelo "sepa" que la cima de una montaña debe ser más fría que el valle, aunque no haya una estación en la cima.
    - **Distancia a la costa:** Ayuda a moderar las temperaturas en zonas costeras.
    - **Datos Satelitales (MODIS):** En versiones recientes, utilizan datos de temperatura superficial terrestre obtenidos por satélite como covariable para mejorar la precisión en áreas con pocas estaciones físicas.

- **Fenomenos del niño y la niña:** Se obtienen a partir de los datos de (...)
- **Datos de trafico en peajes:** Obtenidos de la ANI con metadatos tomados de https://www.colombiaenmapas.gov.co/?u=0&t=19&servicio=100

# Diccionario de datos

- **CodigoMunicipio (DANE):** El codigo DIVIPOLA del municipio.
- **Departamento (SIVIGILA):** El departamento al que pertenece el municipio.
- **Municipio (SIVIGILA):** El nombre del municipio.
- **TipoMunicipio (DANE):** Tipo de municipio ('municipio', 'isla', 'area no municipalizada')
- **Longitud (DANE):** La longitud del municipio.
- **Latitud (DANE):** La latitud del municipio.
- **Year (SIVIGILA):** Año de los registros.
- **Mes (SIVIGILA):** Mes de los registros.
- **CasosConfirmados (SIVIGILA):** Número de casos confirmados de Dengue y Dengue Grave.
- **Elevacion (WorldClim):** Elevación en metros sobre el nivel del mar.
- **Precipitacion (WorldClim):** Precipitación total en mm.
- **TempMax (WorldClim):** Temperatura máxima en grados Celsius.
- **TempMin (WorldClim):** Temperatura mínima en grados Celsius.
- **Fenomeno ():** Se agrega marcacion categorica respecto alas temepraturas del pacifico que son las que afectan a Colombia.

### Instrucciones para construcción de la base de datos a partir de notebook Grupo 6 - BaseDengue.ipynb

Se debe crear un entorno en python con la versión 3.12 e instalar las librerías contenidas en el archivo requirements.txt, este archivo también contiene los comandos para llevar a cabo toda la instalación siempre y cuando se tenga conda o miniconda instalado.


1. Los archivos de base están todos en el comprimido `input_col.rar` que contiene 6 archivos y se encuentra en el sharepoint en la carpeta Data/input_col:
    - **Base_deng.xlsx:** Que contiene los casos confirmados a nivel municipal en colombia extraidos del SISPRO, contiene en su hoja `res` los casos confirmados por municipio de residencia y su hoja `ocur` los casos confirmados por municipio de ocurrencia.
    - **Listados_DIVIPOLA.xlsx:** Archivo del DANE que contiene las coordenadas, codigo y tipo de municipio se obtiene al descargar en el link mencionado más arriba en la tabla *Listado completos de Codificación Divipola*.
    - **nino34.long.anom.csv:** Contiene el cambio en temperatura mensual en el pacifico, denota cuando hay cambios que pueden implicar un fenomenro del niño y de la niña.
    - **Tráfico_Vehicular_ANI_20260411.csv:** Contiene la información de trafico por peaje anivel nacional y con datos hitoricos de 2014 a 2025.
    - **Peajes_.csv:** Metadatos de los peajes a nivel nacional con ubicación por coordenadas.
    - **anexo_pobreza_monetaria_20_departamento.xls:** Se tienen datos de pobreza monetaria para 2012-2020 a nivel departamental, adicional tiene una serie de 2002 a 2020 con el coeficiente GINI por departamento.

2. La data de WorldClim se obtiene a partir de multiples archivos, 3 archivos comprimidos por cada variable y se alamacenan en data\worldclim, allí se descomprimen cada uno en su carpeta correspondiente.
4. Con los archivos ubicados basta con ejecutar todo el notebook y se generan los siguientes archivos:

    - **dengue_data_v2_{res/cour}.csv:** Contiene la data base de casos confirmados de Dengue junto con la información de datos climaticos y fenomenos del niño y de la niña, los reportes vacios de años están presentes con dato vacio en casos confirmados.

    - **df_monthly_trafico_peajes.csv:** Contiene el trafico mensual por peaje con coordenadas del peaje, la idea es que sirva de insumo para tomar trafico de peajes relenates para el muncipio.

    - **municipios_con_peajes_cercanos.csv:** Contiene la relación de municipio y peajes a 25, 50 y 75 km. Aún no sé usa porque se encuntra un numero importante de vacios y no es clara la relacion de trafico a municipio pues pese a mostrar el dato del trafico no se sabe si este es de alguna froma uniforme o no.

    - **pobreza_departamental.csv:** Datos del coeficiente del GINI departamental 2002 a 2020. No se añade aún por ventana de tiempo.

    - **dengue_data_v3_{res/cour}.csv:** Esta versión incluye los rezagos en variables cliamticas, no se ha hace tratemiento de datos faltantes aún.

    - **dengue_data_v4_{res/cour}.csv:** En esta versión se eliminan variables identificadoras (municipio, departamento, pais, tipo municipio, etc) y se tiene el tratamiento de imputación para valores faltantes, además se ha acotado el DF a solo los departamentos con mayor calidad de datos.

    - **dengue_data_v5_{res/cour}.csv:** Es la versión que tiene la data lista para usarse en analisis de correlación y ejecución del modelo.

    - **dengue_data_vfnl_{res/cour}.csv:** Contiene los datos listos para ejecutar el modelo depués de habe rhecho limpieza relacionada al análisis de correlación.

    
    