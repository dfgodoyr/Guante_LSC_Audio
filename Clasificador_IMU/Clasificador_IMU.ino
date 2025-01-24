/*
  IMU Classifier
  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.
  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.
  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.
  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry
  This example code is in the public domain.
*/

#include "Arduino_BMI270_BMM150.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

const float accelerationThreshold = 2; // threshold of significant in G's
const int numSamples = 70;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
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

const int flexPins[5] = {A0, A1, A2, A3, A4};  // Pines analógicos de los 5 flex sensors
// Ajustar si tu placa tiene 3.3 V o 5 V
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

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // (Opcional, si tu placa lo soporta: SAMD21, Due, etc.)
  analogReadResolution(ADC_BITS);

  // Configurar pines como entrada para flex sensores
  for(int i = 0; i < 5; i++) {
    pinMode(flexPins[i], INPUT);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // wait for significant motion
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        samplesRead = 0;
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      float angles[5];

      for(int i = 0; i < 5; i++) {
        int ADCflex = analogRead(flexPins[i]);
        float Vflex = ADCflex * (VCC / ADC_MAX);
        float angle = 0.0;

        if(Vflex > 0) {
          float Rflex = R_DIV[i] * (VCC / Vflex - 1.0);
          angle = map(Rflex, flatResistance[i], bendResistance[i], 0, 90.0);
          if(angle < 0) angle = 0;
        }
        angles[i] = angle;
      }

      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
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
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

        // Loop through the output tensor values from the model
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