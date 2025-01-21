# README de la Aplicación

## Descripción

Esta aplicación está diseñada para generar un Dataset de forma automática dado un tema en concreto, genera el número que marques de subtemas y proporciona tanto preguntas como respuestas de los temas que indiques. El texto es creado usando Facebook/llama-3.1-405b

Tiene un sistema de filtrado de calidad de las respuestas usando nvidia/nemotron-4-340b-reward donde a través de distintos valores se premia a la opción mejor valorada.

Para usarlo debes utilizar API de Nvidia NIM

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd tu_repositorio
   ```
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar la aplicación, utiliza el siguiente comando:
```bash
python app.py
```

Asegúrate de que el cliente y la lista de preguntas estén correctamente configurados en el archivo `app.py`.

## Ejemplo de Código

En el archivo `app.py`, se utiliza el siguiente código para generar respuestas:

```python
question_response_list = response_generator(client, question_list_formatted)
```

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
