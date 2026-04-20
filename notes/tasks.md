# Aufgabenverteilung – AIC Team

---

## Richard & Paul – FoundationPose

**Ziel:**
Robuste 6D-Poseschätzung (Position + Rotation) des Kabel-Ports aus Kameradaten, um der Policy präzise Zielposen zu liefern.

**Schritte:**
1. FoundationPose in der Simulationsumgebung (Gazebo) einrichten
2. 3D-Modell des Kabel-Ports als Referenz einbinden
3. Pipeline aufbauen: RGB-Bild + Tiefenbild → 6D-Pose
4. Ausgabe als ROS 2 Topic für die Policy verfügbar machen
5. Genauigkeit validieren anhand bekannter Ground-Truth-Posen aus der Simulation

**Technologien:**
- [FoundationPose](https://github.com/NVlabs/FoundationPose) (NVidia)
- ROS 2 (Topic-Publisher für Pose-Output)
- Python, PyTorch
- Gazebo (Simulationsumgebung)

---

## Moritz – Mono zu Stereo / Tiefenschätzung

**Ziel:**
Synthetische Tiefeninformation aus dem monokularen RGB-Kamerabild der Simulation erzeugen, da FoundationPose ein Tiefenbild als Input benötigt.

**Schritte:**
1. Verfügbare Monocular Depth Estimation Modelle evaluieren (z.B. Depth Anything v2, MiDaS)
2. Modell in der Simulationsumgebung testen und Qualität beurteilen
3. Ausgabe als synthetisches Tiefenbild im Format das FoundationPose erwartet
4. Alternativ: Stereo-Setup in Gazebo konfigurieren für echtes Tiefenbild

**Technologien:**
- [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) oder MiDaS
- Google Colab (T4 GPU) für erste Tests
- ROS 2 (sensor_msgs/Image für Depth-Output)

---

## Bernhard – Task Description / Port-Erkennung

**Ziel:**
Den richtigen Ziel-Port in der Szene identifizieren basierend auf dem Action Goal der Engine, und diese Information als Eingabe für die Greifplanung bereitstellen.

**Schritte:**
1. Szenenstruktur und verfügbare Ports aus den Challenge-Docs analysieren
2. Erkennungslogik implementieren: welcher Port entspricht dem Goal der Engine?
3. Visuelle Detektion des Ports im Kamerabild (z.B. mit YOLO oder DINO-Features)
4. Ausgabe: Port-ID + grobe Position als Input für FoundationPose-Pipeline

**Technologien:**
- YOLO (schnelle Objekterkennung) oder DINO (robuste Vision-Embeddings)
- ROS 2 (Action Goal Interface: `/insert_cable`)
- Python
- aic_interfaces (Challenge-eigene ROS 2 Message-Definitionen)
