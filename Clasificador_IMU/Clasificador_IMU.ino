#include "Arduino_BMI270_BMM150.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema_schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

const float accelerationThreshold = 2; // Umbral significativo en G's
const int numSamples = 70;

int samplesRead = numSamples;

// Variables globales utilizadas para TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// Cargar todas las operaciones de TFLM. Puedes eliminar esta línea y 
// solo cargar las operaciones necesarias para reducir el tamaño 
// del sketch compilado.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Crear un búfer de memoria estática para TFLM. Es posible que 
// necesite ajustarse según el modelo que estés utilizando.
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Array para mapear el índice de los gestos a un nombre
const char* GESTURES[] = {
    "ayuda",
    "comoestas",
    "cuanto",
    "donde",
    "explicar",
    "gracias",
    "hola",
    "perdon",
    "porfavor",
    "quepaso"
};

const int flexPins[5] = {A0, A1, A2, A3, A4};  // Pines analógicos de los 5 sensores flexibles
// Ajustar si tu placa funciona con 3.3 V o 5 V
const float VCC = 3.3;

// Valores de divisor de tensión para cada sensor (R_DIV)
float R_DIV[5] = {
  47000.0, // para sensor 1
  47000.0, // para sensor 2
  47000.0, // para sensor 3
  47000.0, // para sensor 4
  47000.0  // para sensor 5
};

// Valores de resistencias en posición plana (flat) y doblada (bend) para cada sensor
float flatResistance[5] = {
  29000.0, // plano sensor 1
  28000.0, // plano sensor 2
  25000.0, // plano sensor 3
  34000.0, // plano sensor 4
  31000.0  // plano sensor 5
};

float bendResistance[5] = {
  100000.0, // doblado sensor 1
  100000.0, // doblado sensor 2
  100000.0, // doblado sensor 3
  100000.0, // doblado sensor 4
  100000.0  // doblado sensor 5
};

// Configuración de resolución ADC (depende de la placa)
const int ADC_BITS = 12;                
const int ADC_MAX  = (1 << ADC_BITS) - 1; // 4095 para 12 bits

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Inicializar el IMU
  if (!IMU.begin()) {
    Serial.println("¡Error al inicializar el IMU!");
    while (1);
  }

  // (Opcional, si tu placa lo soporta: SAMD21, Due, etc.)
  analogReadResolution(ADC_BITS);

  // Configurar pines como entrada para sensores flexibles
  for (int i = 0; i < 5; i++) {
    pinMode(flexPins[i], INPUT);
  }

  // Mostrar las tasas de muestreo del acelerómetro y el giroscopio
  Serial.print("Tasa de muestreo del acelerómetro = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Tasa de muestreo del giroscopio = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // Obtener la representación TFL del modelo en el arreglo de bytes
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("¡Incompatibilidad de versión del modelo!");
    while (1);
  }

  // Crear un intérprete para ejecutar el modelo
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Asignar memoria para los tensores de entrada y salida del modelo
  tflInterpreter->AllocateTensors();

  // Obtener punteros para los tensores de entrada y salida del modelo
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Esperar un movimiento significativo
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // Leer los datos de aceleración
      IMU.readAcceleration(aX, aY, aZ);

      // Sumar los valores absolutos
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // Verificar si está por encima del umbral
      if (aSum >= accelerationThreshold) {
        // Reiniciar el conteo de muestras leídas
        samplesRead = 0;
        break;
      }
    }
  }

  // Verificar si se han leído todas las muestras requeridas desde
  // la última vez que se detectó un movimiento significativo
  while (samplesRead < numSamples) {
    // Verificar si hay nuevos datos de aceleración Y giroscopio disponibles
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // Leer los datos de aceleración y giroscopio
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      float angles[5];

      for (int i = 0; i < 5; i++) {
        int ADCflex = analogRead(flexPins[i]);
        float Vflex = ADCflex * (VCC / ADC_MAX);
        float angle = 0.0;

        if (Vflex > 0) {
          float Rflex = R_DIV[i] * (VCC / Vflex - 1.0);
          angle = map(Rflex, flatResistance[i], bendResistance[i], 0, 90.0);
          if (angle < 0) angle = 0;
        }
        angles[i] = angle;
      }

      // Normalizar los datos del IMU entre 0 y 1 y almacenarlos en el tensor
      tflInputTensor->data.f[samplesRead * 11 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 11 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 11 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 11 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 11 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 11 + 5] = (gZ + 2000.0) / 4000.0;
      for (int i = 0; i < 5; i++) {
        float normalizedAngle = angles[i] / 90.0f; // 0..1
        tflInputTensor->data.f[samplesRead * 11 + 6 + i] = normalizedAngle;
      }

      samplesRead++;

      if (samplesRead == numSamples) {
        // Ejecutar inferencia
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("¡Fallo en la invocación!");
          while (1);
          return;
        }

        // Iterar a través de los valores del tensor de salida del modelo
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6);
        }
        Serial.println();
      }
    }
  }
}
