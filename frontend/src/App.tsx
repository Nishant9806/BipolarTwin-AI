import { useEffect, useState } from "react";
import {
  Activity,
  HeartPulse,
  Moon,
  Footprints,
  Brain,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import "./App.css";

interface DashboardData {
  patient_id: string;
  digital_twin: {
    last_observation_date: string;
    current_recorded_state: string | null;
    latest_sleep_hours: number | null;
    latest_steps: number | null;
    latest_resting_hr: number | null;
    latest_hrv_rmssd: number | null;
  };
  ehr_records: {
    profile: Record<string, string | number>;
    medications: Array<Record<string, string | number>>;
    labs: Array<Record<string, string | number>>;
  };
  personal_baseline: Record<string, number>;
  baseline_deviations: Record<string, number>;
  current_mood: {
    predicted_class: string;
    probabilities: Record<string, number>;
  };
  future_outcome: {
    target: string;
    estimated_probability: number;
    validation_status: string;
  };
  limitations: string[];
}

function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchDashboard = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(
        "http://127.0.0.1:8000/api/patient/P001"
      );

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const result: DashboardData = await response.json();
      setData(result);
    } catch {
      setError(
        "Unable to connect to the Digital Twin API. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchDashboard();
  }, []);

  if (loading) {
    return (
      <main className="loading-screen">
        <Activity size={42} />
        <h2>Loading BipolarTwin AI...</h2>
        <p>Connecting to the Digital Twin engine</p>
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="loading-screen">
        <AlertTriangle size={42} />
        <h2>Connection Error</h2>
        <p>{error || "Dashboard data is unavailable."}</p>
        <button onClick={fetchDashboard}>
          <RefreshCw size={16} /> Retry Connection
        </button>
      </main>
    );
  }

  const twin = data.digital_twin;
  const mood = data.current_mood;
  const future = data.future_outcome;

  const formatValue = (
    value: number | null | undefined,
    suffix = ""
  ) => {
    if (value === null || value === undefined) return "N/A";
    return `${value}${suffix}`;
  };

  const formatMood = (value: string) =>
    value.charAt(0).toUpperCase() + value.slice(1);

  const moodColors: Record<string, string> = {
    stable: "#16a085",
    depression: "#6478d3",
    mania: "#e6a23c",
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Brain size={25} />
          </div>
          <div>
            <h2>BipolarTwin</h2>
            <span>AI HEALTH PLATFORM</span>
          </div>
        </div>

        <nav>
          <div 
            className={`nav-item ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
          >
            <Activity size={19} />
            Overview
          </div>
          <div 
            className={`nav-item ${activeTab === "digital-twin" ? "active" : ""}`}
            onClick={() => setActiveTab("digital-twin")}
          >
            <HeartPulse size={19} />
            Digital Twin
          </div>
          <div 
            className={`nav-item ${activeTab === "mood-analysis" ? "active" : ""}`}
            onClick={() => setActiveTab("mood-analysis")}
          >
            <Brain size={19} />
            Mood Analysis
          </div>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          Backend connection
          <strong>Experimental</strong>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">DIGITAL HEALTH MONITORING</p>
            <h1>Patient Overview</h1>
            <p className="subtitle">
              Monitor longitudinal data and experimental model outputs.
            </p>
          </div>

          <button className="refresh-button" onClick={fetchDashboard}>
            <RefreshCw size={16} />
            Refresh Data
          </button>
        </header>

        {activeTab === "overview" && (
          <>
            <section className="patient-banner">
              <div className="patient-avatar">
                <Brain size={28} />
              </div>

              <div className="patient-details">
                <span>MONITORED PROFILE</span>
                <h2>Patient {data.patient_id}</h2>
                <p>Bipolar disorder digital twin · Research prototype</p>
              </div>

              <div className="record-date">
                <span>LAST OBSERVATION</span>
                <strong>{twin.last_observation_date}</strong>
              </div>
            </section>

            <section className="section-heading">
              <div>
                <h2>Current Health Indicators</h2>
                <p>Latest recorded wearable measurements</p>
              </div>
            </section>

            <section className="metrics-grid">
              <MetricCard
                icon={<Moon size={21} />}
                label="Sleep Duration"
                value={formatValue(twin.latest_sleep_hours, " hrs")}
                detail={`30-day baseline: ${data.personal_baseline.sleep_hours?.toFixed(1) ?? "N/A"} hrs`}
              />

              <MetricCard
                icon={<Footprints size={21} />}
                label="Daily Steps"
                value={
                  twin.latest_steps?.toLocaleString("en-IN") ?? "N/A"
                }
                detail={`30-day baseline: ${data.personal_baseline.steps?.toLocaleString("en-IN") ?? "N/A"}`}
              />

              <MetricCard
                icon={<HeartPulse size={21} />}
                label="Resting Heart Rate"
                value={formatValue(twin.latest_resting_hr, " bpm")}
                detail={`30-day baseline: ${data.personal_baseline.resting_hr?.toFixed(1) ?? "N/A"} bpm`}
              />

              <MetricCard
                icon={<Activity size={21} />}
                label="Heart Rate Variability"
                value={formatValue(twin.latest_hrv_rmssd, " ms")}
                detail={`30-day baseline: ${data.personal_baseline.hrv_rmssd?.toFixed(1) ?? "N/A"} ms`}
              />
            </section>

            <section className="analysis-grid">
              <div className="panel mood-panel">
                <div className="panel-header">
                  <div>
                    <h2>Current Mood Classification</h2>
                    <p>Logistic Regression model output</p>
                  </div>
                  <Brain size={22} />
                </div>

                <div
                  className="mood-badge"
                  style={{
                    color: moodColors[mood.predicted_class] || "#6478d3",
                  }}
                >
                  {formatMood(mood.predicted_class)}
                </div>

                <p className="small-label">MODEL PROBABILITIES</p>

                {Object.entries(mood.probabilities).map(
                  ([label, probability]) => (
                    <div className="probability-row" key={label}>
                      <div className="probability-label">
                        <span>{formatMood(label)}</span>
                        <strong>
                          {(probability * 100).toFixed(1)}%
                        </strong>
                      </div>

                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${Math.max(0, Math.min(100, probability * 100))}%`,
                            backgroundColor: moodColors[label] || "#6478d3",
                          }}
                        />
                      </div>
                    </div>
                  )
                )}

                <p className="disclaimer">
                  Experimental model output, not a clinical diagnosis.
                </p>
              </div>

              <div className="panel forecast-panel">
                <div className="panel-header">
                  <div>
                    <h2>Future Outcome Estimate</h2>
                    <p>Experimental forecasting model</p>
                  </div>
                  <Activity size={22} />
                </div>

                <span className="forecast-label">
                  {future.target.replaceAll("_", " ").toUpperCase()}
                </span>

                <div className="forecast-value">
                  {(future.estimated_probability * 100).toFixed(1)}
                  <span>%</span>
                </div>

                <div className="progress-track forecast-track">
                  <div
                    className="progress-fill forecast-fill"
                    style={{
                      width: `${Math.max(0, Math.min(100, future.estimated_probability * 100))}%`,
                    }}
                  />
                </div>

                <div className="forecast-status">
                  <span className="status-dot warning-dot" />
                  Validation: {future.validation_status}
                </div>

                <div className="notice">
                  <AlertTriangle size={19} />
                  <p>
                    This is an experimental estimate for the dataset's
                    recorded target. Its clinical meaning and predictive
                    reliability have not been established.
                  </p>
                </div>
              </div>
            </section>

            <section className="panel baseline-panel">
              <div className="panel-header">
                <div>
                  <h2>Personal Baseline Deviations</h2>
                  <p>Latest measurements compared with the 30-day baseline</p>
                </div>
              </div>

              <div className="deviation-grid">
                {Object.entries(data.baseline_deviations).map(
                  ([label, value]) => (
                    <div className="deviation-item" key={label}>
                      <span>{label.replaceAll("_", " ")}</span>
                      <strong className={value < 0 ? "negative" : "positive"}>
                        {value > 0 ? "+" : ""}
                        {value.toFixed(2)}
                      </strong>
                    </div>
                  )
                )}
              </div>

              <p className="disclaimer">
                Deviations are descriptive comparisons, not evidence of a
                mood episode or a medical warning.
              </p>
            </section>

            <section className="panel limitations-panel">
              <h2>Research Limitations</h2>
              {data.limitations.map((limitation) => (
                <p className="limitation-item" key={limitation}>
                  <AlertTriangle size={16} />
                  {limitation}
                </p>
              ))}
            </section>
          </>
        )}

        {activeTab === "digital-twin" && (
          <>
            <section className="section-heading">
              <div>
                <h2>Digital Twin Medical Profile</h2>
                <p>Current simulated state and patient records.</p>
              </div>
            </section>
            
            <div className="analysis-grid">
              <div className="panel">
                <div className="panel-header">
                  <div>
                    <h2>Latest Telemetry State</h2>
                    <p>Current wearable metrics</p>
                  </div>
                  <Activity size={22} />
                </div>
                <div className="deviation-grid" style={{ marginTop: "1.5rem" }}>
                  <div className="deviation-item">
                    <span>Last Sync</span>
                    <strong>{twin.last_observation_date}</strong>
                  </div>
                  <div className="deviation-item">
                    <span>Sleep Duration</span>
                    <strong>{formatValue(twin.latest_sleep_hours, " hrs")}</strong>
                  </div>
                  <div className="deviation-item">
                    <span>Daily Steps</span>
                    <strong>{twin.latest_steps?.toLocaleString("en-IN") ?? "N/A"}</strong>
                  </div>
                  <div className="deviation-item">
                    <span>Resting HR</span>
                    <strong>{formatValue(twin.latest_resting_hr, " bpm")}</strong>
                  </div>
                  <div className="deviation-item">
                    <span>HRV (RMSSD)</span>
                    <strong>{formatValue(twin.latest_hrv_rmssd, " ms")}</strong>
                  </div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <div>
                    <h2>Electronic Health Records</h2>
                    <p>Clinical history and medications</p>
                  </div>
                  <HeartPulse size={22} />
                </div>
                
                {data.ehr_records ? (
                  <div style={{ marginTop: "1.5rem" }}>
                    <p className="small-label" style={{ marginBottom: "0.5rem" }}>RECENT MEDICATIONS</p>
                    <div style={{ overflowX: "auto", marginBottom: "1.5rem" }}>
                      {data.ehr_records.medications.length > 0 ? (
                        <table style={{ width: "100%", fontSize: "0.85rem", textAlign: "left", borderCollapse: "collapse" }}>
                          <thead>
                            <tr style={{ borderBottom: "1px solid #30363d", color: "#8b949e" }}>
                              {Object.keys(data.ehr_records.medications[0])
                                .filter(k => k !== "patient_id")
                                .map(k => <th key={k} style={{ padding: "0.5rem", textTransform: "capitalize" }}>{k.replaceAll("_", " ")}</th>)}
                            </tr>
                          </thead>
                          <tbody>
                            {data.ehr_records.medications.slice(-5).reverse().map((row, i) => (
                              <tr key={i} style={{ borderBottom: "1px solid #21262d" }}>
                                {Object.keys(row)
                                  .filter(k => k !== "patient_id")
                                  .map(k => <td key={k} style={{ padding: "0.5rem", color: "#c9d1d9" }}>{String(row[k])}</td>)}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <p style={{ fontSize: "0.85rem", color: "#8b949e" }}>No medication records found.</p>
                      )}
                    </div>

                    <p className="small-label" style={{ marginBottom: "0.5rem" }}>RECENT LAB RESULTS</p>
                    <div style={{ overflowX: "auto" }}>
                      {data.ehr_records.labs.length > 0 ? (
                        <table style={{ width: "100%", fontSize: "0.85rem", textAlign: "left", borderCollapse: "collapse" }}>
                          <thead>
                            <tr style={{ borderBottom: "1px solid #30363d", color: "#8b949e" }}>
                              {Object.keys(data.ehr_records.labs[0])
                                .filter(k => k !== "patient_id")
                                .map(k => <th key={k} style={{ padding: "0.5rem", textTransform: "capitalize" }}>{k.replaceAll("_", " ")}</th>)}
                            </tr>
                          </thead>
                          <tbody>
                            {data.ehr_records.labs.slice(-5).reverse().map((row, i) => (
                              <tr key={i} style={{ borderBottom: "1px solid #21262d" }}>
                                {Object.keys(row)
                                  .filter(k => k !== "patient_id")
                                  .map(k => <td key={k} style={{ padding: "0.5rem", color: "#c9d1d9" }}>{String(row[k])}</td>)}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <p style={{ fontSize: "0.85rem", color: "#8b949e" }}>No lab records found.</p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="notice" style={{ marginTop: "1.5rem" }}>
                    <AlertTriangle size={24} />
                    <p>
                      <strong>EHR Data Not Found in API Response.</strong>
                    </p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {activeTab === "mood-analysis" && (
          <>
            <section className="section-heading">
              <div>
                <h2>Predictive Mood Analysis</h2>
                <p>Current classification and 7-day clinical forecast.</p>
              </div>
            </section>

            <section className="analysis-grid">
              <div className="panel mood-panel">
                <div className="panel-header">
                  <div>
                    <h2>Current Mood Classification</h2>
                    <p>Logistic Regression model output</p>
                  </div>
                  <Brain size={22} />
                </div>

                <div
                  className="mood-badge"
                  style={{
                    color: moodColors[mood.predicted_class] || "#6478d3",
                  }}
                >
                  {formatMood(mood.predicted_class)}
                </div>

                <p className="small-label">MODEL PROBABILITIES</p>

                {Object.entries(mood.probabilities).map(
                  ([label, probability]) => (
                    <div className="probability-row" key={label}>
                      <div className="probability-label">
                        <span>{formatMood(label)}</span>
                        <strong>{(probability * 100).toFixed(1)}%</strong>
                      </div>

                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(100, probability * 100)
                            )}%`,
                            backgroundColor: moodColors[label] || "#6478d3",
                          }}
                        />
                      </div>
                    </div>
                  )
                )}

                <p className="disclaimer">
                  Experimental model output, not a clinical diagnosis.
                </p>
              </div>

              <div className="panel forecast-panel">
                <div className="panel-header">
                  <div>
                    <h2>Future Outcome Estimate</h2>
                    <p>Experimental forecasting model</p>
                  </div>
                  <Activity size={22} />
                </div>

                <span className="forecast-label">
                  {future.target.replaceAll("_", " ").toUpperCase()}
                </span>

                <div className="forecast-value">
                  {(future.estimated_probability * 100).toFixed(1)}
                  <span>%</span>
                </div>

                <div className="progress-track forecast-track">
                  <div
                    className="progress-fill forecast-fill"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, future.estimated_probability * 100)
                      )}%`,
                    }}
                  />
                </div>

                <div className="forecast-status">
                  <span className="status-dot warning-dot" />
                  Validation: {future.validation_status}
                </div>

                <div className="notice">
                  <AlertTriangle size={19} />
                  <p>
                    This is an experimental estimate for the dataset's recorded
                    target. Its clinical meaning and predictive reliability have
                    not been established.
                  </p>
                </div>
              </div>
            </section>
          </>
        )}

        <footer>
          BipolarTwin AI · Digital Twin Challenge 2026 · Research
          prototype — not for clinical decision-making.
        </footer>
      </main>
    </div>
  );
}

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
}

function MetricCard({
  icon,
  label,
  value,
  detail,
}: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <p>{label}</p>
      <h3>{value}</h3>
      <span>{detail}</span>
    </div>
  );
}

export default App;