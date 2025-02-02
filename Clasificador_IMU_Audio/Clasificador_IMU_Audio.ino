#include "Arduino_BMI270_BMM150.h"

// Incluir la librería del DFPlayer Mini
#include "DFRobotDFPlayerMini.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

// ----- Configuración del reconocimiento de gestos ----- //
const float accelerationThreshold = 2; // Umbral significativo en G's
const int numSamples = 70;

int samplesRead = numSamples;

// Variables globales para TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Búfer de memoria para TFLM (ajustar según el modelo)
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

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// Pines y configuración para los sensores flexibles
const int flexPins[5] = {A0, A1, A2, A3, A4};  // Pines analógicos
const float VCC = 3.3;  // Voltaje de la placa (ajustar si fuera necesario)

float R_DIV[5] = {
  47000.0, // para sensor 1
  47000.0, // para sensor 2
  47000.0, // para sensor 3
  47000.0, // para sensor 4
  47000.0  // para sensor 5
};

// Valores de resistencia en posición plana y doblada para cada sensor
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

// Configuración del ADC (12 bits)
const int ADC_BITS = 12;                
const int ADC_MAX  = (1 << ADC_BITS) - 1; // 4095 para 12 bits

// ----- Configuración del DFPlayer Mini ----- //
DFRobotDFPlayerMini myDFPlayer;  // Objeto para controlar el DFPlayer

// ------------------------------------------------------ //

void setup() {
  // Inicializar el monitor serial (USB)
  Serial.begin(9600);
  while (!Serial);


  // Inicializar el puerto serial hardware adicional para el DFPlayer (pines 0 y 1)
  Serial1.begin(9600);
  delay(2000);
  //myDFPlayer = new DFRobotDFPlayerMini();

  // Inicializar el DFPlayer Mini
  if (!myDFPlayer.begin(Serial1)) {
    Serial.println("Error al iniciar DFPlayer Mini");
    while (1);
  }
  delay(2000);  // Espera a que el módulo esté listo
  myDFPlayer.volume(30);               // Establece el volumen (0~30)
  myDFPlayer.EQ(DFPLAYER_EQ_NORMAL);     // Ecualizador normal
  myDFPlayer.outputDevice(DFPLAYER_DEVICE_SD);  // Usar la tarjeta SD

  // Inicializar el IMU
  if (!IMU.begin()) {
    Serial.println("¡Error al inicializar el IMU!");
    while (1);
  }

  // Configurar la resolución del ADC
  analogReadResolution(ADC_BITS);

  // Configurar pines de los sensores flexibles
  for (int i = 0; i < 5; i++) {
    pinMode(flexPins[i], INPUT);
  }

  // Mostrar tasas de muestreo del acelerómetro y giroscopio
  Serial.print("Tasa de muestreo del acelerómetro = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Tasa de muestreo del giroscopio = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  Serial.println();

  // Inicializar el modelo TensorFlow Lite
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("¡Incompatibilidad de versión del modelo!");
    while (1);
  }

  // Crear el intérprete para el modelo
  tflInterpreter = new tflite::MicroInterpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  tflInterpreter->AllocateTensors();

  // Obtener punteros para los tensores de entrada y salida
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Esperar a que se detecte un movimiento significativo
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
      if (aSum >= accelerationThreshold) {
        samplesRead = 0;  // Reiniciar el conteo de muestras
        break;
      }
    }
  }

  // Recolectar muestras hasta alcanzar numSamples
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      float angles[5];

      // Leer y procesar cada sensor flexible
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

      // Normalizar y almacenar datos en el tensor de entrada
      int baseIndex = samplesRead * 11;
      tflInputTensor->data.f[baseIndex + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[baseIndex + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[baseIndex + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[baseIndex + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[baseIndex + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[baseIndex + 5] = (gZ + 2000.0) / 4000.0;
      for (int i = 0; i < 5; i++) {
        float normalizedAngle = angles[i] / 90.0f; // Normalizar entre 0 y 1
        tflInputTensor->data.f[baseIndex + 6 + i] = normalizedAngle;
      }

      samplesRead++;

      // Cuando se han leído todas las muestras, se ejecuta la inferencia
      if (samplesRead == numSamples) {
        Serial.println("¡Entrando en la invocación!");
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        Serial.println("¡Saliendo de la invocación!");
        if (invokeStatus != kTfLiteOk) {
          Serial.println("¡Fallo en la invocación!");
          while (1);
          return;
        }

        // Imprimir las probabilidades para cada gesto
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6);
        }
        Serial.println();

        // Determinar el gesto reconocido (el de mayor probabilidad)
        int recognizedGesture = 0;
        float maxProb = tflOutputTensor->data.f[0];
        for (int i = 1; i < NUM_GESTURES; i++) {
          if (tflOutputTensor->data.f[i] > maxProb) {
            maxProb = tflOutputTensor->data.f[i];
            recognizedGesture = i;
          }
        }
        Serial.print("Gesto reconocido: ");
        Serial.println(GESTURES[recognizedGesture]);

        // Reproducir la pista de audio asociada al gesto reconocido.
        // Se asume que los archivos están en la carpeta 1 y que el número
        // de pista es (recognizedGesture + 1)
        myDFPlayer.playFolder(1, recognizedGesture + 1);
        Serial.print("Reproduciendo pista: ");
        Serial.println(recognizedGesture + 1);

        // (Opcional) Agregar un delay para evitar reproducir múltiples veces en gestos consecutivos
        delay(3000);
      }
    }
  }
}
