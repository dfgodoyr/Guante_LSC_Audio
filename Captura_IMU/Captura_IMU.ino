#include "Arduino_BMI270_BMM150.h"

const float VCC = 3.3; // Voltaje VCC de la placa de Arduino 3.3 V para el divisor de tensión

// --- Configuración IMU ---
const float accelerationThreshold = 2; // umbral significativo en G's
const int numSamples = 70;             // cantidad de muestras que se capturan tras detección
int samplesRead = numSamples;

// --- Configuración Flex Sensores ---
const int flexPins[5] = {A0, A1, A2, A3, A4};  // Pines analógicos de los 5 flex sensors

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

void setup() {
  Serial.begin(9600);
  while (!Serial); // Espera a que se inicie Serial (opcional, útil en placas SAMD)

  // Inicializamos el IMU
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

  // Imprimimos encabezado en formato CSV (incluyendo columnas para flex)
  // aX,aY,aZ,gX,gY,gZ,Angle1,Angle2,Angle3,Angle4,Angle5
  Serial.println("aX,aY,aZ,gX,gY,gZ,Angle1,Angle2,Angle3,Angle4,Angle5");
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Espera hasta que haya un movimiento significativo
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      if (aSum >= accelerationThreshold) {
        samplesRead = 0;  // reinicia el contador de muestras
        break;
      }
    }
  }

  // Una vez detectado el movimiento, leemos numSamples de datos
  while (samplesRead < numSamples) {
    // Leemos tanto aceleración como giroscopio si están disponibles
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // --- Leemos los flex sensores y calculamos sus ángulos ---
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
      
      // Imprimimos todo en una misma línea CSV
      // aX, aY, aZ, gX, gY, gZ, Angle1, Angle2, Angle3, Angle4, Angle5
      Serial.print(aX, 3); Serial.print(',');
      Serial.print(aY, 3); Serial.print(',');
      Serial.print(aZ, 3); Serial.print(',');
      Serial.print(gX, 3); Serial.print(',');
      Serial.print(gY, 3); Serial.print(',');
      Serial.print(gZ, 3); 

      // Agregamos los 5 ángulos
      for(int i = 0; i < 5; i++) {
        Serial.print(',');
        Serial.print(angles[i], 1);  // 1 decimal (ajusta a tu gusto)
      }

      Serial.println();

      samplesRead++;

      // Si ya se completaron las muestras, imprime línea en blanco de separación
      if (samplesRead == numSamples) {
        Serial.println();
      }
    }
  }
}
