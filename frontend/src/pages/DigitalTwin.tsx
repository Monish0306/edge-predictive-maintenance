import { useRef, useState, useEffect, Suspense } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Environment } from "@react-three/drei";
import * as THREE from "three";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity, Thermometer, Gauge, Wind,
  AlertTriangle, CheckCircle, Info,
  RotateCcw, ZoomIn, Eye
} from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { simulateReading } from "@/lib/api";

// ── TYPES ──────────────────────────────────────────────
interface SensorData {
  anomaly_probability: number;
  health_score: number;
  severity: string;
  sensor_data: number[];
}

interface ComponentHealth {
  name: string;
  health: number;
  temperature: number;
  status: "normal" | "warning" | "critical";
  description: string;
  sensor_index: number;
}

// ── HEALTH COLOR ────────────────────────────────────────
function getHealthColor(health: number): string {
  if (health >= 80) return "#22C55E";
  if (health >= 60) return "#EAB308";
  if (health >= 40) return "#F97316";
  return "#EF4444";
}

function getHealthColorThree(health: number): THREE.Color {
  if (health >= 80) return new THREE.Color("#22C55E");
  if (health >= 60) return new THREE.Color("#EAB308");
  if (health >= 40) return new THREE.Color("#F97316");
  return new THREE.Color("#EF4444");
}

// ── ENGINE COMPONENTS ───────────────────────────────────

// Fan Blades
function FanBlades({ health, onClick, selected }: {
  health: number; onClick: () => void; selected: boolean
}) {
  const groupRef = useRef<THREE.Group>(null);
  const color = getHealthColorThree(health);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.z += delta * (health / 100) * 3;
    }
  });

  return (
    <group ref={groupRef} position={[-3.5, 0, 0]} onClick={onClick}>
      {[0, 60, 120, 180, 240, 300].map((angle, i) => (
        <mesh
          key={i}
          rotation={[0, 0, (angle * Math.PI) / 180]}
          position={[0, 0, 0]}
        >
          <boxGeometry args={[1.8, 0.15, 0.3]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={selected ? 0.5 : health < 60 ? 0.3 : 0.1}
            metalness={0.8}
            roughness={0.2}
          />
        </mesh>
      ))}
      {/* Hub */}
      <mesh>
        <cylinderGeometry args={[0.3, 0.3, 0.4, 16]} />
        <meshStandardMaterial color="#1E40AF" metalness={0.9} roughness={0.1} />
      </mesh>
    </group>
  );
}

// Engine Body
function EngineBody({ health, onClick, selected }: {
  health: number; onClick: () => void; selected: boolean
}) {
  const color = getHealthColorThree(health);
  return (
    <group position={[0, 0, 0]} onClick={onClick}>
      {/* Main nacelle */}
      <mesh>
        <cylinderGeometry args={[1.1, 1.3, 6, 32]} />
        <meshStandardMaterial
          color={selected ? "#3B82F6" : "#1E293B"}
          emissive={selected ? new THREE.Color("#3B82F6") : color}
          emissiveIntensity={selected ? 0.3 : 0.05}
          metalness={0.9}
          roughness={0.15}
        />
      </mesh>
      {/* Front cone */}
      <mesh position={[-3.2, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <coneGeometry args={[0.9, 1.2, 32]} />
        <meshStandardMaterial color="#0F172A" metalness={0.95} roughness={0.1} />
      </mesh>
      {/* Rear nozzle */}
      <mesh position={[3.3, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <coneGeometry args={[0.7, 1.0, 32]} />
        <meshStandardMaterial
          color={health < 60 ? "#EF444480" : "#0F172A"}
          emissive={health < 60 ? new THREE.Color("#EF4444") : new THREE.Color("#F97316")}
          emissiveIntensity={health < 60 ? 0.6 : 0.2}
          metalness={0.7} roughness={0.3}
        />
      </mesh>
    </group>
  );
}

// Compressor
function Compressor({ health, onClick, selected }: {
  health: number; onClick: () => void; selected: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const color = getHealthColorThree(health);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.5;
    }
  });

  return (
    <group position={[-1.5, 0, 0]} onClick={onClick}>
      <mesh ref={meshRef}>
        <torusGeometry args={[0.7, 0.2, 8, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 0.6 : health < 60 ? 0.4 : 0.15}
          metalness={0.8}
          roughness={0.2}
          wireframe={health < 40}
        />
      </mesh>
    </group>
  );
}

// Combustion Chamber
function CombustionChamber({ health, onClick, selected }: {
  health: number; onClick: () => void; selected: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const time = useRef(0);

  useFrame((_, delta) => {
    time.current += delta;
    if (meshRef.current) {
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0.3 + Math.sin(time.current * 4) * 0.2;
    }
  });

  const color = getHealthColorThree(health);

  return (
    <group position={[0.5, 0, 0]} onClick={onClick}>
      <mesh ref={meshRef}>
        <cylinderGeometry args={[0.6, 0.6, 1.2, 16]} />
        <meshStandardMaterial
          color={selected ? "#3B82F6" : health < 60 ? "#EF4444" : "#F97316"}
          emissive={selected ? new THREE.Color("#3B82F6") : color}
          emissiveIntensity={0.4}
          metalness={0.6}
          roughness={0.4}
        />
      </mesh>
    </group>
  );
}

// Turbine
function Turbine({ health, onClick, selected }: {
  health: number; onClick: () => void; selected: boolean
}) {
  const groupRef = useRef<THREE.Group>(null);
  const color = getHealthColorThree(health);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.z -= delta * (health / 100) * 2;
    }
  });

  return (
    <group ref={groupRef} position={[2.2, 0, 0]} onClick={onClick}>
      {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => (
        <mesh
          key={i}
          rotation={[0, 0, (angle * Math.PI) / 180]}
        >
          <boxGeometry args={[1.0, 0.1, 0.25]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={selected ? 0.5 : health < 60 ? 0.4 : 0.1}
            metalness={0.85}
            roughness={0.15}
          />
        </mesh>
      ))}
      <mesh>
        <cylinderGeometry args={[0.2, 0.2, 0.3, 16]} />
        <meshStandardMaterial color="#1E40AF" metalness={0.9} roughness={0.1} />
      </mesh>
    </group>
  );
}

// Exhaust with animated heat
function Exhaust({ health }: { health: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const time = useRef(0);

  useFrame((_, delta) => {
    time.current += delta;
    if (meshRef.current) {
      meshRef.current.scale.x = 1 + Math.sin(time.current * 3) * 0.05;
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      mat.opacity = 0.4 + Math.sin(time.current * 2) * 0.2;
    }
  });

  return (
    <group position={[4.5, 0, 0]}>
      <mesh ref={meshRef}>
        <coneGeometry args={[0.8, 2, 16]} />
        <meshStandardMaterial
          color={health < 60 ? "#EF4444" : "#F97316"}
          emissive={health < 60 ? new THREE.Color("#EF4444") : new THREE.Color("#F97316")}
          emissiveIntensity={0.8}
          transparent
          opacity={0.5}
        />
      </mesh>
    </group>
  );
}

// Particle Effects for critical status
function CriticalParticles({ active }: { active: boolean }) {
  const pointsRef = useRef<THREE.Points>(null);
  const count = 50;

  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3]     = (Math.random() - 0.5) * 8;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 3;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 2;
  }

  useFrame((_, delta) => {
    if (pointsRef.current && active) {
      pointsRef.current.rotation.y += delta * 0.2;
    }
  });

  if (!active) return null;

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#EF4444"
        size={0.05}
        transparent
        opacity={0.6}
      />
    </points>
  );
}

// Grid floor
function Grid() {
  return (
    <gridHelper
      args={[20, 20, "#1E3A5F", "#0F172A"]}
      position={[0, -2, 0]}
    />
  );
}

// Complete Engine Scene
function EngineScene({
  components, selectedComponent, onSelectComponent
}: {
  components: ComponentHealth[];
  selectedComponent: number | null;
  onSelectComponent: (i: number) => void;
}) {
  const comp = (i: number) => components[i] || { health: 100 };
  const isCritical = components.some(c => c.status === "critical");

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <directionalLight position={[10, 10, 5]} intensity={1.2} castShadow />
      <pointLight position={[-5, 5, 0]} intensity={0.8} color="#3B82F6" />
      <pointLight position={[5, -3, 0]} intensity={0.6} color="#F97316" />
      <spotLight
        position={[0, 8, 0]}
        angle={0.4}
        penumbra={0.5}
        intensity={1}
        color="#60A5FA"
      />

      {/* Engine group - rotate for better view */}
      <group rotation={[0.2, 0, 0]}>
        <EngineBody
          health={comp(1).health}
          onClick={() => onSelectComponent(1)}
          selected={selectedComponent === 1}
        />
        <FanBlades
          health={comp(0).health}
          onClick={() => onSelectComponent(0)}
          selected={selectedComponent === 0}
        />
        <Compressor
          health={comp(2).health}
          onClick={() => onSelectComponent(2)}
          selected={selectedComponent === 2}
        />
        <CombustionChamber
          health={comp(3).health}
          onClick={() => onSelectComponent(3)}
          selected={selectedComponent === 3}
        />
        <Turbine
          health={comp(4).health}
          onClick={() => onSelectComponent(4)}
          selected={selectedComponent === 4}
        />
        <Exhaust health={comp(4).health} />
        <CriticalParticles active={isCritical} />
      </group>

      <Grid />
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={5}
        maxDistance={20}
        autoRotate={selectedComponent === null}
        autoRotateSpeed={0.5}
      />
      <Environment preset="city" />
    </>
  );
}

// ── MAIN DIGITAL TWIN PAGE ──────────────────────────────
export default function DigitalTwin() {
  const [sensorData, setSensorData]           = useState<SensorData | null>(null);
  const [selectedComp, setSelectedComp]       = useState<number | null>(null);
  const [mode, setMode]                       = useState<"normal"|"warning"|"fault">("normal");
  const [autoRefresh, setAutoRefresh]         = useState(true);
  const [loading, setLoading]                 = useState(true);

  // Build component health from sensor data
  const buildComponents = (data: SensorData): ComponentHealth[] => {
    const s = data.sensor_data || Array(15).fill(0.3);
    const prob = data.anomaly_probability;

    return [
      {
        name: "Fan Assembly",
        health: Math.max(0, 100 - (s[1] || 0.3) * 100),
        temperature: 400 + (s[1] || 0.3) * 250,
        status: (s[1] || 0) > 0.7 ? "critical" : (s[1] || 0) > 0.5 ? "warning" : "normal",
        description: "Fan inlet temperature & speed monitoring. Detects blade wear and bearing failures.",
        sensor_index: 1,
      },
      {
        name: "Engine Nacelle",
        health: Math.max(0, 100 - prob * 100),
        temperature: 350 + prob * 200,
        status: prob > 0.7 ? "critical" : prob > 0.4 ? "warning" : "normal",
        description: "Overall engine casing structural integrity and thermal envelope.",
        sensor_index: 0,
      },
      {
        name: "HPC Compressor",
        health: Math.max(0, 100 - (s[3] || 0.3) * 100),
        temperature: 550 + (s[3] || 0.3) * 350,
        status: (s[3] || 0) > 0.7 ? "critical" : (s[3] || 0) > 0.5 ? "warning" : "normal",
        description: "High-pressure compressor. Key indicator for HPC degradation fault mode.",
        sensor_index: 3,
      },
      {
        name: "Combustion Chamber",
        health: Math.max(0, 100 - (s[6] || 0.3) * 100),
        temperature: 1200 + (s[6] || 0.3) * 600,
        status: (s[6] || 0) > 0.7 ? "critical" : (s[6] || 0) > 0.5 ? "warning" : "normal",
        description: "Combustion efficiency and fuel flow ratio. High temps indicate fuel system issues.",
        sensor_index: 6,
      },
      {
        name: "LPT Turbine",
        health: Math.max(0, 100 - (s[4] || 0.3) * 100),
        temperature: 800 + (s[4] || 0.3) * 400,
        status: (s[4] || 0) > 0.7 ? "critical" : (s[4] || 0) > 0.5 ? "warning" : "normal",
        description: "Low-pressure turbine outlet temperature. Degradation causes efficiency loss.",
        sensor_index: 4,
      },
    ];
  };

  const [components, setComponents] = useState<ComponentHealth[]>(
    buildComponents({
      anomaly_probability: 0,
      health_score: 100,
      severity: "NORMAL",
      sensor_data: Array(15).fill(0.3),
    })
  );

  // Fetch data
  const fetchData = async () => {
    try {
      const data = await simulateReading(mode, 1);
      setSensorData(data);
      setComponents(buildComponents(data as any));
      setLoading(false);
    } catch (e) {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [mode]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [autoRefresh, mode]);

  const selected = selectedComp !== null ? components[selectedComp] : null;

  const statusIcon = (status: string) => {
    if (status === "critical") return <AlertTriangle className="w-4 h-4 text-red-400" />;
    if (status === "warning") return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
    return <CheckCircle className="w-4 h-4 text-green-400" />;
  };

  return (
    <div className="h-full">
      <PageHeader
        title="Digital Twin Simulator"
        subtitle="Interactive 3D engine model with real-time sensor data mapping"
      />

      {/* Controls */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <select
          value={mode}
          onChange={e => setMode(e.target.value as any)}
          className="bg-card border border-border text-sm rounded-lg px-3 py-2 text-foreground"
        >
          <option value="normal">Normal Operation</option>
          <option value="warning">Warning Mode</option>
          <option value="fault">Fault Simulation</option>
        </select>

        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium border transition-all ${
            autoRefresh
              ? "bg-green-500/20 border-green-500/30 text-green-400"
              : "bg-card border-border text-muted-foreground"
          }`}
        >
          <Activity className="w-4 h-4" />
          {autoRefresh ? "Live" : "Paused"}
        </button>

        <button
          onClick={() => setSelectedComp(null)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm border border-border bg-card text-muted-foreground hover:text-foreground transition-all"
        >
          <RotateCcw className="w-4 h-4" />
          Reset View
        </button>

        <button
          onClick={fetchData}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm border border-border bg-card text-muted-foreground hover:text-foreground transition-all"
        >
          <Eye className="w-4 h-4" />
          Refresh
        </button>

        {/* Overall health badge */}
        {sensorData && (
          <div
            className="ml-auto px-4 py-2 rounded-lg border text-sm font-bold font-mono"
            style={{
              color: getHealthColor(sensorData.health_score),
              borderColor: getHealthColor(sensorData.health_score) + "40",
              background: getHealthColor(sensorData.health_score) + "10",
            }}
          >
            Engine Health: {sensorData.health_score.toFixed(1)}%
          </div>
        )}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">

        {/* 3D Canvas - takes 3 columns */}
        <div className="lg:col-span-3">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="rounded-xl border border-border bg-card overflow-hidden"
            style={{ height: "520px" }}
          >
            {loading ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="w-12 h-12 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                  <p className="text-muted-foreground text-sm">Loading 3D Engine...</p>
                </div>
              </div>
            ) : (
              <>
                <Canvas
                  camera={{ position: [0, 3, 12], fov: 50 }}
                  style={{ background: "transparent" }}
                  shadows
                >
                  <Suspense fallback={null}>
                    <EngineScene
                      components={components}
                      selectedComponent={selectedComp}
                      onSelectComponent={setSelectedComp}
                    />
                  </Suspense>
                </Canvas>

                {/* Overlay instructions */}
                <div className="absolute bottom-3 left-3 flex gap-2 pointer-events-none">
                  <span className="text-[10px] text-slate-500 bg-black/40 px-2 py-1 rounded">
                    🖱️ Drag to rotate
                  </span>
                  <span className="text-[10px] text-slate-500 bg-black/40 px-2 py-1 rounded">
                    🔍 Scroll to zoom
                  </span>
                  <span className="text-[10px] text-slate-500 bg-black/40 px-2 py-1 rounded">
                    👆 Click part to inspect
                  </span>
                </div>
              </>
            )}
          </motion.div>
        </div>

        {/* Right Panel */}
        <div className="space-y-3">

          {/* Selected component detail */}
          <AnimatePresence mode="wait">
            {selected ? (
              <motion.div
                key="selected"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-xl border bg-card p-4"
                style={{
                  borderColor: getHealthColor(selected.health) + "40",
                  background: getHealthColor(selected.health) + "08",
                }}
              >
                <div className="flex items-center gap-2 mb-3">
                  {statusIcon(selected.status)}
                  <h3 className="font-bold text-sm text-white">{selected.name}</h3>
                </div>

                {/* Health bar */}
                <div className="mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted-foreground">Health</span>
                    <span className="font-mono font-bold" style={{ color: getHealthColor(selected.health) }}>
                      {selected.health.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${selected.health}%` }}
                      transition={{ duration: 0.8 }}
                      className="h-full rounded-full"
                      style={{ background: getHealthColor(selected.health) }}
                    />
                  </div>
                </div>

                {/* Temperature */}
                <div className="flex items-center gap-2 mb-3 p-2 rounded-lg bg-black/20">
                  <Thermometer className="w-4 h-4 text-orange-400" />
                  <div>
                    <div className="text-xs text-muted-foreground">Temperature</div>
                    <div className="text-sm font-mono font-bold text-orange-400">
                      {selected.temperature.toFixed(0)}°F
                    </div>
                  </div>
                </div>

                {/* Status */}
                <div className={`px-3 py-2 rounded-lg text-xs font-bold uppercase tracking-wider text-center mb-3 ${
                  selected.status === "critical"
                    ? "bg-red-500/20 text-red-400 border border-red-500/30"
                    : selected.status === "warning"
                    ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
                    : "bg-green-500/20 text-green-400 border border-green-500/30"
                }`}>
                  {selected.status}
                </div>

                {/* Description */}
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {selected.description}
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="hint"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="rounded-xl border border-border bg-card p-4 text-center"
              >
                <div className="text-3xl mb-2">👆</div>
                <p className="text-sm text-muted-foreground">
                  Click any engine component to inspect its health
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* All components list */}
          <div className="rounded-xl border border-border bg-card p-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-3">
              Component Status
            </h3>
            <div className="space-y-2">
              {components.map((comp, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedComp(i)}
                  className={`w-full flex items-center gap-3 p-2 rounded-lg transition-all hover:bg-white/5 ${
                    selectedComp === i ? "bg-blue-500/10 border border-blue-500/20" : ""
                  }`}
                >
                  {statusIcon(comp.status)}
                  <div className="flex-1 text-left">
                    <div className="text-xs font-medium text-white truncate">
                      {comp.name}
                    </div>
                    <div className="w-full h-1 bg-slate-800 rounded-full mt-1">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${comp.health}%`,
                          background: getHealthColor(comp.health),
                        }}
                      />
                    </div>
                  </div>
                  <span
                    className="text-xs font-mono font-bold"
                    style={{ color: getHealthColor(comp.health) }}
                  >
                    {comp.health.toFixed(0)}%
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Temperature heatmap legend */}
          <div className="rounded-xl border border-border bg-card p-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-3">
              Health Legend
            </h3>
            <div className="space-y-2">
              {[
                { label: "Healthy (80-100%)", color: "#22C55E" },
                { label: "Warning (60-80%)", color: "#EAB308" },
                { label: "Degraded (40-60%)", color: "#F97316" },
                { label: "Critical (0-40%)", color: "#EF4444" },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: item.color, boxShadow: `0 0 6px ${item.color}` }}
                  />
                  <span className="text-xs text-muted-foreground">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom sensor readings */}
      {sensorData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-4 rounded-xl border border-border bg-card p-5"
        >
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Gauge className="w-4 h-4 text-blue-400" />
            Live Sensor Readings — All 15 Sensors
          </h3>
          <div className="grid grid-cols-5 md:grid-cols-8 lg:grid-cols-15 gap-2">
            {(sensorData.sensor_data || []).map((val, i) => (
              <div key={i} className="text-center">
                <div className="text-[9px] text-muted-foreground mb-1">S{i + 1}</div>
                <div
                  className="h-16 rounded-lg relative overflow-hidden"
                  style={{ background: "#0F172A" }}
                >
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: `${val * 100}%` }}
                    transition={{ duration: 0.5 }}
                    className="absolute bottom-0 w-full rounded-lg"
                    style={{
                      background: val > 0.7
                        ? "#EF4444"
                        : val > 0.5
                        ? "#F97316"
                        : val > 0.3
                        ? "#EAB308"
                        : "#22C55E",
                      opacity: 0.8,
                    }}
                  />
                </div>
                <div
                  className="text-[9px] font-mono mt-1"
                  style={{
                    color: val > 0.7 ? "#EF4444" : val > 0.5 ? "#F97316" : "#94A3B8"
                  }}
                >
                  {(val * 100).toFixed(0)}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}