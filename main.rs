use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{PrimaryWindow, WindowPlugin},
};
use bevy_rapier3d::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        mpsc::{self, Receiver},
        Mutex,
    },
};

// ============================================================================
// CONSTANTS
// ============================================================================

const LEFT_PANEL: f32 = 250.0;
const RIGHT_PANEL: f32 = 300.0;

const LEG_NAMES: [&str; 4] = ["FL", "FR", "BL", "BR"];

const JOINT_ORDER: [&str; 12] = [
    "FL_hip_yaw", "FL_hip_pitch", "FL_knee",
    "FR_hip_yaw", "FR_hip_pitch", "FR_knee",
    "BL_hip_yaw", "BL_hip_pitch", "BL_knee",
    "BR_hip_yaw", "BR_hip_pitch", "BR_knee",
];

// ============================================================================
// CONFIG
// ============================================================================

#[derive(Clone, Deserialize, Resource)]
struct RobotConfig {
    control_hz: f32,
    physics_hz: f32,
    physics_substeps: usize,
    gait: GaitConfig,
    chassis: ChassisConfig,
    links: LinkConfig,
}

#[derive(Clone, Deserialize)]
struct GaitConfig {
    step_period_s: f32,
    max_climb_height_m: f32,
}

#[derive(Clone, Deserialize)]
struct ChassisConfig {
    length_m: f32,
    width_m: f32,
    height_m: f32,
    mass_kg: f32,
}

#[derive(Clone, Deserialize)]
struct LinkConfig {
    coxa_length_m: f32,
    femur_length_m: f32,
    tibia_length_m: f32,
    link_radius_m: f32,
    coxa_mass_kg: f32,
    femur_mass_kg: f32,
    tibia_mass_kg: f32,
    foot_radius_m: f32,
}

// ============================================================================
// ROBOT VISUAL MODEL
// ============================================================================

#[derive(Component)]
struct RobotBody;

#[derive(Component)]
struct RobotPart;

#[derive(Clone, Copy)]
enum VisualPart {
    ServoA,
    ServoB,
    LinkAB,
    LinkBFoot,
    Foot,
}

#[derive(Component)]
struct LegVisual {
    leg: &'static str,
    part: VisualPart,
}

#[derive(Resource)]
struct RobotPose {
    position: Vec3,
    heading: f32,
    base_height: f32,
    climb_offset: f32,
    previous_position: Vec3,
    measured_speed: f32,
}

impl Default for RobotPose {
    fn default() -> Self {
        let p = Vec3::new(0.0, 0.16, -0.85);
        Self {
            position: p,
            heading: 0.0,
            base_height: 0.16,
            climb_offset: 0.0,
            previous_position: p,
            measured_speed: 0.0,
        }
    }
}

// ============================================================================
// WORLD
// ============================================================================

#[derive(Component)]
struct Ground;

#[derive(Component)]
struct Obstacle {
    radius: f32,
    height: f32,
    climbable: bool,
}

#[derive(Component)]
struct GoalMarker;

#[derive(Component)]
struct MainCamera;

#[derive(Resource)]
struct WorldState {
    goal: Vec3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlacementMode {
    None,
    Obstacle,
    ClimbStep,
    Goal,
}

#[derive(Resource)]
struct PlacementState {
    mode: PlacementMode,
}

impl Default for PlacementState {
    fn default() -> Self {
        Self {
            mode: PlacementMode::None,
        }
    }
}

#[derive(Resource, Default)]
struct DragState {
    obstacle: Option<Entity>,
}

// ============================================================================
// CONTROL
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum ControlMode {
    Manual,
    Automatic,
}

#[derive(Resource)]
struct ControlState {
    mode: ControlMode,
    manual_forward: f32,
    manual_turn: f32,
}

impl Default for ControlState {
    fn default() -> Self {
        Self {
            mode: ControlMode::Manual,
            manual_forward: 0.0,
            manual_turn: 0.0,
        }
    }
}

#[derive(Resource)]
struct ActionState {
    joint_targets: [f32; 12],
    forward: f32,
    turn: f32,
    active_leg: String,
    gait_phase: String,
    climb_mode: bool,
    controller: String,
}

impl Default for ActionState {
    fn default() -> Self {
        Self {
            joint_targets: [0.0; 12],
            forward: 0.0,
            turn: 0.0,
            active_leg: "FR".to_string(),
            gait_phase: "STOP".to_string(),
            climb_mode: false,
            controller: "waiting".to_string(),
        }
    }
}

#[derive(Resource)]
struct ControlClock {
    elapsed: f32,
    reset_pending: bool,
}

impl Default for ControlClock {
    fn default() -> Self {
        Self {
            elapsed: 0.0,
            reset_pending: true,
        }
    }
}

// ============================================================================
// CAMERA
// ============================================================================

#[derive(Resource)]
struct CameraRig {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            focus: Vec3::new(0.0, 0.10, 0.35),
            yaw: 2.48,
            pitch: 0.56,
            distance: 2.65,
        }
    }
}

// ============================================================================
// PYTHON BRIDGE
// ============================================================================

struct BridgeIo {
    _child: Child,
    stdin: ChildStdin,
    receiver: Receiver<ActionPacket>,
}

#[derive(Resource)]
struct PythonBridge {
    io: Mutex<BridgeIo>,
}

#[derive(Serialize)]
struct ObstacleObservation {
    x: f32,
    z: f32,
    radius: f32,
    height: f32,
    climbable: bool,
}

#[derive(Serialize)]
struct NavigationObservation {
    goal_local: [f32; 2],
    obstacles: Vec<ObstacleObservation>,
}

#[derive(Serialize)]
struct ControlObservation {
    mode: &'static str,
    forward: f32,
    turn: f32,
}

#[derive(Serialize)]
struct ObservationPacket {
    dt: f32,
    episode_reset: bool,
    joint_angles_rad: [f32; 12],
    joint_velocities_rad_s: [f32; 12],
    chassis_quat_xyzw: [f32; 4],
    chassis_linear_velocity_m_s: [f32; 3],
    chassis_angular_velocity_rad_s: [f32; 3],
    foot_contacts: [bool; 4],
    control: ControlObservation,
    navigation: NavigationObservation,
}

#[derive(Deserialize)]
struct ActionPacket {
    action_joint_targets_rad: Vec<f32>,
    #[serde(default)]
    forward_command: f32,
    #[serde(default)]
    turn_command: f32,
    #[serde(default)]
    active_leg: String,
    #[serde(default)]
    gait_phase: String,
    #[serde(default)]
    climb_mode: bool,
    #[serde(default)]
    controller: String,
    #[serde(default)]
    error: Option<String>,
}

// ============================================================================
// UI
// ============================================================================

#[derive(Clone, Copy)]
enum SettingKey {
    ABLength,
    BFootLength,
    LinkWidth,
    BodyWidth,
    BodyLength,
    BodyMass,
    ABMass,
    BFootMass,
}

#[derive(Component, Clone, Copy)]
enum UiAction {
    Manual,
    Automatic,
    Forward,
    Backward,
    Left,
    Right,
    Stop,
    ResetRobot,
    ResetCamera,
    AddObstacle,
    AddClimbStep,
    ChangeGoal,
    ClearObstacles,
    Adjust(SettingKey, f32),
}

#[derive(Component)]
struct TelemetryText;

#[derive(Component)]
struct PlacementText;

#[derive(Component)]
struct SettingText(SettingKey);

// ============================================================================
// VISUAL MATERIALS
// ============================================================================

#[derive(Resource, Clone)]
struct Materials {
    body: Handle<StandardMaterial>,
    servo_a: Handle<StandardMaterial>,
    servo_b: Handle<StandardMaterial>,
    link: Handle<StandardMaterial>,
    foot: Handle<StandardMaterial>,
    ground: Handle<StandardMaterial>,
    obstacle: Handle<StandardMaterial>,
    climb: Handle<StandardMaterial>,
    goal: Handle<StandardMaterial>,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let config_text = fs::read_to_string("robot_config.json")
        .expect("robot_config.json must be beside Cargo.toml");

    let config: RobotConfig =
        serde_json::from_str(&config_text).expect("Invalid robot_config.json");

    let bridge = start_python_bridge()
        .expect("Could not start python3 movement_policy.py --bridge");

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.72, 0.80, 0.88)))
        .insert_resource(TimestepMode::Fixed {
            dt: 1.0 / config.physics_hz.max(60.0),
            substeps: config.physics_substeps.max(1),
        })
        .insert_resource(config)
        .insert_resource(bridge)
        .insert_resource(ControlState::default())
        .insert_resource(ActionState::default())
        .insert_resource(ControlClock::default())
        .insert_resource(RobotPose::default())
        .insert_resource(CameraRig::default())
        .insert_resource(PlacementState::default())
        .insert_resource(DragState::default())
        .insert_resource(WorldState {
            goal: Vec3::new(0.55, 0.02, 1.70),
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "SpiderBot Rescue Simulator".to_string(),
                resolution: (1440, 900).into(),
                resizable: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                keyboard_controls,
                ui_buttons,
                world_mouse,
                camera_mouse,
                send_controller_observation,
                receive_controller_action,
                move_robot,
                update_robot_visual,
                update_camera,
                update_ui,
                update_button_colors,
                draw_world_debug,
            )
                .chain(),
        )
        .run();
}

// ============================================================================
// SETUP
// ============================================================================

fn setup(
    mut commands: Commands,
    config: Res<RobotConfig>,
    world: Res<WorldState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    let mats = Materials {
        body: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.05, 0.35, 0.82),
            metallic: 0.15,
            perceptual_roughness: 0.38,
            ..default()
        }),
        servo_a: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.95, 0.22, 0.16),
            metallic: 0.05,
            perceptual_roughness: 0.45,
            ..default()
        }),
        servo_b: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(1.00, 0.66, 0.08),
            metallic: 0.05,
            perceptual_roughness: 0.45,
            ..default()
        }),
        link: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.18, 0.22, 0.27),
            metallic: 0.45,
            perceptual_roughness: 0.35,
            ..default()
        }),
        foot: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.10, 0.12, 0.15),
            ..default()
        }),
        ground: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.48, 0.53, 0.56),
            perceptual_roughness: 0.95,
            ..default()
        }),
        obstacle: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.34, 0.36, 0.39),
            perceptual_roughness: 0.90,
            ..default()
        }),
        climb: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.52, 0.31, 0.12),
            perceptual_roughness: 0.85,
            ..default()
        }),
        goal: standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.08, 0.86, 0.22),
            emissive: Color::srgb(0.02, 0.20, 0.04).into(),
            ..default()
        }),
    };

    commands.insert_resource(mats.clone());

    // Ground is deliberately bright enough to make the robot obvious.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(5.5, 0.05, 5.5))),
        MeshMaterial3d(mats.ground.clone()),
        Transform::from_xyz(0.0, -0.025, 0.0),
        RigidBody::Fixed,
        Collider::cuboid(2.75, 0.025, 2.75),
        Ground,
    ));

    // Robot: one clear body + four visible two-servo legs.
    spawn_robot_visual(
        &mut commands,
        &mut meshes,
        &mats,
        &config,
    );

    // Demo obstacles.
    spawn_obstacle(
        &mut commands,
        &mut meshes,
        &mats,
        Vec3::new(-0.25, 0.0, 0.00),
        0.12,
        0.18,
        false,
    );
    spawn_obstacle(
        &mut commands,
        &mut meshes,
        &mats,
        Vec3::new(0.24, 0.0, 0.48),
        0.12,
        0.20,
        false,
    );
    spawn_obstacle(
        &mut commands,
        &mut meshes,
        &mats,
        Vec3::new(-0.20, 0.0, 0.92),
        0.10,
        0.16,
        false,
    );

    // Brown low box = climb experiment.
    spawn_obstacle(
        &mut commands,
        &mut meshes,
        &mats,
        Vec3::new(0.10, 0.0, 1.28),
        0.14,
        0.065,
        true,
    );

    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.06, 0.04))),
        MeshMaterial3d(mats.goal.clone()),
        Transform::from_translation(world.goal),
        GoalMarker,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(
            Quat::from_euler(EulerRot::XYZ, -0.85, -0.55, 0.0)
        ),
    ));

    commands.spawn((
        PointLight {
            intensity: 1200.0,
            range: 8.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-1.0, 2.0, -1.4),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(1.30, 1.50, -1.50)
            .looking_at(Vec3::new(0.0, 0.12, 0.35), Vec3::Y),
        MainCamera,
    ));

    setup_ui(&mut commands);
}

fn spawn_robot_visual(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    mats: &Materials,
    config: &RobotConfig,
) {
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(
            config.chassis.width_m,
            config.chassis.height_m,
            config.chassis.length_m,
        ))),
        MeshMaterial3d(mats.body.clone()),
        Transform::default(),
        RigidBody::KinematicPositionBased,
        Collider::cuboid(
            config.chassis.width_m / 2.0,
            config.chassis.height_m / 2.0,
            config.chassis.length_m / 2.0,
        ),
        RobotBody,
        RobotPart,
    ));

    let servo_size = 0.030;
    let link_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let servo_mesh = meshes.add(Cuboid::new(servo_size, servo_size, servo_size));
    let foot_mesh = meshes.add(Sphere::new(config.links.foot_radius_m));

    for leg in LEG_NAMES {
        commands.spawn((
            Mesh3d(servo_mesh.clone()),
            MeshMaterial3d(mats.servo_a.clone()),
            Transform::default(),
            LegVisual {
                leg,
                part: VisualPart::ServoA,
            },
            RobotPart,
        ));

        commands.spawn((
            Mesh3d(servo_mesh.clone()),
            MeshMaterial3d(mats.servo_b.clone()),
            Transform::default(),
            LegVisual {
                leg,
                part: VisualPart::ServoB,
            },
            RobotPart,
        ));

        commands.spawn((
            Mesh3d(link_mesh.clone()),
            MeshMaterial3d(mats.link.clone()),
            Transform::default(),
            LegVisual {
                leg,
                part: VisualPart::LinkAB,
            },
            RobotPart,
        ));

        commands.spawn((
            Mesh3d(link_mesh.clone()),
            MeshMaterial3d(mats.link.clone()),
            Transform::default(),
            LegVisual {
                leg,
                part: VisualPart::LinkBFoot,
            },
            RobotPart,
        ));

        commands.spawn((
            Mesh3d(foot_mesh.clone()),
            MeshMaterial3d(mats.foot.clone()),
            Transform::default(),
            LegVisual {
                leg,
                part: VisualPart::Foot,
            },
            RobotPart,
        ));
    }
}

fn spawn_obstacle(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    mats: &Materials,
    ground_pos: Vec3,
    radius: f32,
    height: f32,
    climbable: bool,
) {
    let material = if climbable {
        mats.climb.clone()
    } else {
        mats.obstacle.clone()
    };

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(
            radius * 2.0,
            height,
            radius * 2.0,
        ))),
        MeshMaterial3d(material),
        Transform::from_xyz(
            ground_pos.x,
            height / 2.0,
            ground_pos.z,
        ),
        RigidBody::KinematicPositionBased,
        Collider::cuboid(radius, height / 2.0, radius),
        Obstacle {
            radius,
            height,
            climbable,
        },
    ));
}

// ============================================================================
// PYTHON BRIDGE
// ============================================================================

fn start_python_bridge() -> std::io::Result<PythonBridge> {
    let mut child = Command::new("python3")
        .args(["movement_policy.py", "--bridge"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdin = child.stdin.take().expect("Python stdin unavailable");
    let stdout = child.stdout.take().expect("Python stdout unavailable");

    let (tx, rx) = mpsc::channel::<ActionPacket>();

    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };

            match serde_json::from_str::<ActionPacket>(&line) {
                Ok(packet) => {
                    if tx.send(packet).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    eprintln!("Controller JSON error: {err}");
                    eprintln!("Python output: {line}");
                }
            }
        }
    });

    Ok(PythonBridge {
        io: Mutex::new(BridgeIo {
            _child: child,
            stdin,
            receiver: rx,
        }),
    })
}

fn send_controller_observation(
    time: Res<Time>,
    config: Res<RobotConfig>,
    control: Res<ControlState>,
    world: Res<WorldState>,
    pose: Res<RobotPose>,
    action: Res<ActionState>,
    mut clock: ResMut<ControlClock>,
    bridge: Res<PythonBridge>,
    obstacles: Query<(&Transform, &Obstacle)>,
) {
    clock.elapsed += time.delta_secs();

    let period = 1.0 / config.control_hz.max(1.0);

    if clock.elapsed < period {
        return;
    }

    let dt = clock.elapsed;
    clock.elapsed = 0.0;

    let right = Vec3::new(
        pose.heading.cos(),
        0.0,
        -pose.heading.sin(),
    );

    let forward = Vec3::new(
        pose.heading.sin(),
        0.0,
        pose.heading.cos(),
    );

    let goal_delta = world.goal - pose.position;

    let goal_local = [
        goal_delta.dot(right),
        goal_delta.dot(forward),
    ];

    let mut obstacle_obs = Vec::new();

    for (tf, obstacle) in obstacles.iter() {
        let delta = tf.translation - pose.position;

        if delta.length() > 1.6 {
            continue;
        }

        obstacle_obs.push(ObstacleObservation {
            x: delta.dot(right),
            z: delta.dot(forward),
            radius: obstacle.radius,
            height: obstacle.height,
            climbable: obstacle.climbable,
        });
    }

    let control_obs = match control.mode {
        ControlMode::Manual => ControlObservation {
            mode: "manual",
            forward: control.manual_forward,
            turn: control.manual_turn,
        },
        ControlMode::Automatic => ControlObservation {
            mode: "automatic",
            forward: 0.0,
            turn: 0.0,
        },
    };

    let q = Quat::from_rotation_y(pose.heading);

    let packet = ObservationPacket {
        dt,
        episode_reset: clock.reset_pending,
        joint_angles_rad: action.joint_targets,
        joint_velocities_rad_s: [0.0; 12],
        chassis_quat_xyzw: [q.x, q.y, q.z, q.w],
        chassis_linear_velocity_m_s: [
            forward.x * pose.measured_speed,
            0.0,
            forward.z * pose.measured_speed,
        ],
        chassis_angular_velocity_rad_s: [
            0.0,
            action.turn * 0.60,
            0.0,
        ],
        foot_contacts: [true, true, true, true],
        control: control_obs,
        navigation: NavigationObservation {
            goal_local,
            obstacles: obstacle_obs,
        },
    };

    clock.reset_pending = false;

    if let Ok(line) = serde_json::to_string(&packet) {
        if let Ok(mut io) = bridge.io.lock() {
            if writeln!(io.stdin, "{line}").is_ok() {
                let _ = io.stdin.flush();
            }
        }
    }
}

fn receive_controller_action(
    bridge: Res<PythonBridge>,
    mut action: ResMut<ActionState>,
) {
    let Ok(io) = bridge.io.lock() else {
        return;
    };

    while let Ok(packet) = io.receiver.try_recv() {
        if let Some(err) = packet.error {
            error!("movement_policy.py: {err}");
        }

        if packet.action_joint_targets_rad.len() == 12 {
            for i in 0..12 {
                action.joint_targets[i] =
                    packet.action_joint_targets_rad[i];
            }
        }

        action.forward = packet.forward_command;
        action.turn = packet.turn_command;
        action.climb_mode = packet.climb_mode;

        if !packet.active_leg.is_empty() {
            action.active_leg = packet.active_leg;
        }

        if !packet.gait_phase.is_empty() {
            action.gait_phase = packet.gait_phase;
        }

        if !packet.controller.is_empty() {
            action.controller = packet.controller;
        }
    }
}

// ============================================================================
// ROBOT MOVEMENT
// ============================================================================

// IMPORTANT:
// The obstacle and robot-body queries both access Transform.
// `Without<...>` makes the entity sets explicitly disjoint for Bevy ECS,
// preventing runtime error B0001.
fn move_robot(
    time: Res<Time>,
    config: Res<RobotConfig>,
    action: Res<ActionState>,
    obstacles: Query<(&Transform, &Obstacle), Without<RobotBody>>,
    mut pose: ResMut<RobotPose>,
    mut body: Query<&mut Transform, (With<RobotBody>, Without<Obstacle>)>,
) {
    let dt = time.delta_secs().min(0.05);

    // Slow, plausible crawl speed for a small hobby-servo quadruped.
    let max_forward_speed = 0.085;
    let max_reverse_speed = 0.050;
    let turn_speed = 0.62; // rad/s ≈ 35.5 deg/s

    pose.heading += action.turn.clamp(-1.0, 1.0) * turn_speed * dt;

    let forward = Vec3::new(
        pose.heading.sin(),
        0.0,
        pose.heading.cos(),
    );

    let speed = if action.forward >= 0.0 {
        max_forward_speed * action.forward.clamp(0.0, 1.0)
    } else {
        max_reverse_speed * action.forward.clamp(-1.0, 0.0)
    };

    let proposed = pose.position + forward * speed * dt;

    let body_radius =
        (config.chassis.width_m * 0.5)
            .hypot(config.chassis.length_m * 0.5)
            + 0.02;

    let mut blocked = false;
    let mut climb_height: f32 = 0.0;

    for (tf, obstacle) in obstacles.iter() {
        let d = Vec2::new(
            proposed.x - tf.translation.x,
            proposed.z - tf.translation.z,
        )
        .length();

        if obstacle.climbable
            && obstacle.height <= config.gait.max_climb_height_m
        {
            if d < body_radius + obstacle.radius + 0.08 {
                climb_height = climb_height.max(obstacle.height);
            }
            continue;
        }

        if d < body_radius + obstacle.radius {
            blocked = true;
        }
    }

    if !blocked {
        pose.position.x = proposed.x;
        pose.position.z = proposed.z;
    }

    let target_climb = if action.climb_mode {
        climb_height * 0.70
    } else {
        0.0
    };

    let smoothing = 1.0 - (-dt * 6.0).exp();
    pose.climb_offset +=
        (target_climb - pose.climb_offset) * smoothing;

    pose.position.y = pose.base_height + pose.climb_offset;

    let moved = Vec2::new(
        pose.position.x - pose.previous_position.x,
        pose.position.z - pose.previous_position.z,
    )
    .length();

    let instantaneous = moved / dt.max(0.001);

    pose.measured_speed +=
        (instantaneous - pose.measured_speed)
            * (1.0 - (-dt * 5.0).exp());

    pose.previous_position = pose.position;

    for mut tf in body.iter_mut() {
        tf.translation = pose.position;
        tf.rotation = Quat::from_rotation_y(pose.heading);
    }
}

// ============================================================================
// LEG VISUAL ANIMATION
// ============================================================================

fn update_robot_visual(
    config: Res<RobotConfig>,
    pose: Res<RobotPose>,
    action: Res<ActionState>,
    mut legs: Query<(&LegVisual, &mut Transform), Without<RobotBody>>,
) {
    let body_rot = Quat::from_rotation_y(pose.heading);

    let ab_length = config.links.coxa_length_m.max(0.035);
    let bfoot_length =
        (config.links.femur_length_m + config.links.tibia_length_m * 0.35)
            .clamp(0.075, 0.190);

    let thickness =
        (config.links.link_radius_m * 1.8).clamp(0.012, 0.045);

    for (visual, mut tf) in legs.iter_mut() {
        let (side, front) = leg_signs(visual.leg);

        let joint_base = joint_base_index(visual.leg);

        let hip_yaw =
            action.joint_targets[joint_base];

        let hip_pitch =
            action.joint_targets[joint_base + 1];

        let knee =
            action.joint_targets[joint_base + 2];

        let hip_local = Vec3::new(
            side * config.chassis.width_m * 0.54,
            0.0,
            front * config.chassis.length_m * 0.36,
        );

        let outward = Vec3::new(side, 0.0, 0.0);

        // Servo A = horizontal swing/yaw.
        let yaw_dir =
            Quat::from_rotation_y(-side * hip_yaw) * outward;

        let b_local =
            hip_local + yaw_dir * ab_length;

        // Servo B = lift/pitch. Knee target contributes additional flex,
        // but visually we keep one clear B->Foot segment.
        let neutral_down = 58.0_f32.to_radians();

        let down_angle =
            (neutral_down + hip_pitch - knee * 0.55)
                .clamp(18.0_f32.to_radians(), 78.0_f32.to_radians());

        let horizontal = bfoot_length * down_angle.cos();
        let vertical = bfoot_length * down_angle.sin();

        let foot_local =
            b_local + yaw_dir * horizontal + Vec3::NEG_Y * vertical;

        let body_origin = pose.position;

        let a_world =
            body_origin + body_rot * hip_local;
        let b_world =
            body_origin + body_rot * b_local;
        let foot_world =
            body_origin + body_rot * foot_local;

        match visual.part {
            VisualPart::ServoA => {
                tf.translation = a_world;
                tf.rotation = body_rot;
                tf.scale = Vec3::ONE;
            }

            VisualPart::ServoB => {
                tf.translation = b_world;
                tf.rotation = body_rot;
                tf.scale = Vec3::ONE;
            }

            VisualPart::LinkAB => {
                *tf = segment_transform(
                    a_world,
                    b_world,
                    thickness,
                );
            }

            VisualPart::LinkBFoot => {
                *tf = segment_transform(
                    b_world,
                    foot_world,
                    thickness * 0.85,
                );
            }

            VisualPart::Foot => {
                tf.translation = foot_world;
                tf.rotation = Quat::IDENTITY;
                tf.scale = Vec3::ONE;
            }
        }
    }
}

fn leg_signs(leg: &str) -> (f32, f32) {
    match leg {
        "FL" => (-1.0, 1.0),
        "FR" => (1.0, 1.0),
        "BL" => (-1.0, -1.0),
        _ => (1.0, -1.0),
    }
}

fn joint_base_index(leg: &str) -> usize {
    match leg {
        "FL" => 0,
        "FR" => 3,
        "BL" => 6,
        _ => 9,
    }
}

fn segment_transform(
    start: Vec3,
    end: Vec3,
    thickness: f32,
) -> Transform {
    let dir = end - start;
    let length = dir.length().max(0.001);

    Transform {
        translation: (start + end) * 0.5,
        rotation: Quat::from_rotation_arc(
            Vec3::Z,
            dir / length,
        ),
        scale: Vec3::new(
            thickness,
            thickness,
            length,
        ),
    }
}

// ============================================================================
// KEYBOARD
// ============================================================================

fn keyboard_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut control: ResMut<ControlState>,
    mut placement: ResMut<PlacementState>,
    mut camera: ResMut<CameraRig>,
    mut pose: ResMut<RobotPose>,
    mut clock: ResMut<ControlClock>,
    obstacles: Query<Entity, With<Obstacle>>,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::KeyW) {
        set_manual(&mut control, 1.0, 0.0);
    }

    if keys.just_pressed(KeyCode::KeyS) {
        set_manual(&mut control, -0.65, 0.0);
    }

    // Corrected:
    // LEFT = negative turn
    if keys.just_pressed(KeyCode::KeyA) {
        set_manual(&mut control, 0.0, -1.0);
    }

    // RIGHT = positive turn
    if keys.just_pressed(KeyCode::KeyD) {
        set_manual(&mut control, 0.0, 1.0);
    }

    if keys.just_pressed(KeyCode::Space) {
        set_manual(&mut control, 0.0, 0.0);
    }

    if keys.just_pressed(KeyCode::KeyM) {
        control.mode = ControlMode::Automatic;
    }

    if keys.just_pressed(KeyCode::KeyN) {
        set_manual(&mut control, 0.0, 0.0);
    }

    if keys.just_pressed(KeyCode::KeyR) {
        reset_robot_pose(&mut pose);
        clock.reset_pending = true;
        set_manual(&mut control, 0.0, 0.0);
    }

    if keys.just_pressed(KeyCode::KeyF) {
        *camera = CameraRig::default();
    }

    if keys.just_pressed(KeyCode::KeyO) {
        placement.mode = PlacementMode::Obstacle;
    }

    if keys.just_pressed(KeyCode::KeyB) {
        placement.mode = PlacementMode::ClimbStep;
    }

    if keys.just_pressed(KeyCode::KeyG) {
        placement.mode = PlacementMode::Goal;
    }

    if keys.just_pressed(KeyCode::KeyC) {
        for e in obstacles.iter() {
            commands.entity(e).despawn();
        }
    }
}

fn set_manual(
    control: &mut ControlState,
    forward: f32,
    turn: f32,
) {
    control.mode = ControlMode::Manual;
    control.manual_forward = forward;
    control.manual_turn = turn;
}

fn reset_robot_pose(pose: &mut RobotPose) {
    pose.position = Vec3::new(0.0, pose.base_height, -0.85);
    pose.previous_position = pose.position;
    pose.heading = 0.0;
    pose.climb_offset = 0.0;
    pose.measured_speed = 0.0;
}

// ============================================================================
// CAMERA
// ============================================================================

fn camera_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut rig: ResMut<CameraRig>,
) {
    if buttons.pressed(MouseButton::Right) {
        rig.yaw -= motion.delta.x * 0.006;
        rig.pitch =
            (rig.pitch - motion.delta.y * 0.005)
                .clamp(0.18, 1.30);
    }

    if buttons.pressed(MouseButton::Middle) {
        let right =
            Vec3::new(rig.yaw.cos(), 0.0, -rig.yaw.sin());

        let forward =
            Vec3::new(rig.yaw.sin(), 0.0, rig.yaw.cos());

        let scale = rig.distance * 0.0014;

        rig.focus +=
            -right * motion.delta.x * scale
            + forward * motion.delta.y * scale;
    }

    if scroll.delta.y.abs() > 0.001 {
        rig.distance *= (-scroll.delta.y * 0.22).exp();
        rig.distance = rig.distance.clamp(0.55, 4.5);
    }
}

fn update_camera(
    rig: Res<CameraRig>,
    mut cameras: Query<&mut Transform, With<MainCamera>>,
) {
    let Some(mut camera) = cameras.iter_mut().next() else {
        return;
    };

    let cp = rig.pitch.cos();

    let from_focus = Vec3::new(
        rig.yaw.sin() * cp,
        rig.pitch.sin(),
        rig.yaw.cos() * cp,
    );

    camera.translation =
        rig.focus + from_focus * rig.distance;

    camera.translation.y =
        camera.translation.y.max(0.25);

    camera.look_at(rig.focus, Vec3::Y);
}

// ============================================================================
// WORLD MOUSE
// ============================================================================

fn ground_point(
    window: &Window,
    camera: &Camera,
    camera_tf: &GlobalTransform,
) -> Option<Vec3> {
    let cursor = window.cursor_position()?;

    if cursor.x < LEFT_PANEL
        || cursor.x > window.width() - RIGHT_PANEL
    {
        return None;
    }

    let ray =
        camera.viewport_to_world(camera_tf, cursor).ok()?;

    ray.plane_intersection_point(
        Vec3::ZERO,
        InfinitePlane3d::new(Vec3::Y),
    )
}

fn world_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut placement: ResMut<PlacementState>,
    mut drag: ResMut<DragState>,
    mut world: ResMut<WorldState>,
    mut goal_q: Query<
        &mut Transform,
        (With<GoalMarker>, Without<Obstacle>),
    >,
    mut obstacles: Query<
        (Entity, &mut Transform, &Obstacle),
        Without<GoalMarker>,
    >,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mats: Res<Materials>,
) {
    let Some(window) = windows.iter().next() else {
        return;
    };

    let Some((camera, camera_tf)) =
        camera_q.iter().next()
    else {
        return;
    };

    let Some(point) =
        ground_point(window, camera, camera_tf)
    else {
        if buttons.just_released(MouseButton::Left) {
            drag.obstacle = None;
        }
        return;
    };

    if buttons.just_pressed(MouseButton::Left) {
        match placement.mode {
            PlacementMode::Goal => {
                world.goal = Vec3::new(point.x, 0.02, point.z);

                if let Some(mut tf) =
                    goal_q.iter_mut().next()
                {
                    tf.translation = world.goal;
                }

                placement.mode = PlacementMode::None;
                return;
            }

            PlacementMode::Obstacle => {
                spawn_obstacle(
                    &mut commands,
                    &mut meshes,
                    &mats,
                    point,
                    0.12,
                    0.18,
                    false,
                );

                placement.mode = PlacementMode::None;
                return;
            }

            PlacementMode::ClimbStep => {
                spawn_obstacle(
                    &mut commands,
                    &mut meshes,
                    &mats,
                    point,
                    0.14,
                    0.065,
                    true,
                );

                placement.mode = PlacementMode::None;
                return;
            }

            PlacementMode::None => {}
        }

        let mut best: Option<(Entity, f32)> = None;

        for (entity, tf, obstacle) in obstacles.iter_mut() {
            let d = Vec2::new(
                point.x - tf.translation.x,
                point.z - tf.translation.z,
            )
            .length();

            if d <= obstacle.radius + 0.10
                && best
                    .map(|(_, old)| d < old)
                    .unwrap_or(true)
            {
                best = Some((entity, d));
            }
        }

        drag.obstacle = best.map(|v| v.0);
    }

    if buttons.pressed(MouseButton::Left) {
        if let Some(selected) = drag.obstacle {
            if let Ok((_, mut tf, obstacle)) =
                obstacles.get_mut(selected)
            {
                tf.translation.x = point.x;
                tf.translation.z = point.z;
                tf.translation.y = obstacle.height / 2.0;
            }
        }
    }

    if buttons.just_released(MouseButton::Left) {
        drag.obstacle = None;
    }
}

// ============================================================================
// UI SETUP
// ============================================================================

fn setup_ui(commands: &mut Commands) {
    // LEFT PANEL - clean light background.
    commands
        .spawn((
            Node {
                width: px(LEFT_PANEL),
                height: percent(100),
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(0),
                flex_direction: FlexDirection::Column,
                row_gap: px(6),
                padding: UiRect::all(px(12)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.94, 0.96, 0.98)),
            ZIndex(20),
        ))
        .with_children(|panel| {
            heading(panel, "SPIDERBOT CONTROL");
            muted(panel, "W Forward   S Backward\nA LEFT      D RIGHT\nSpace Stop\nM Auto      N Manual");

            section(panel, "MODE");
            button(panel, "Manual", UiAction::Manual);
            button(panel, "Automatic / Potential Field", UiAction::Automatic);

            section(panel, "MOVEMENT");
            button(panel, "W  Forward", UiAction::Forward);
            button(panel, "S  Backward", UiAction::Backward);
            button(panel, "A  LEFT", UiAction::Left);
            button(panel, "D  RIGHT", UiAction::Right);
            button(panel, "STOP", UiAction::Stop);

            section(panel, "WORLD");
            button(panel, "Reset Robot", UiAction::ResetRobot);
            button(panel, "Reset Camera", UiAction::ResetCamera);
            button(panel, "Add Obstacle", UiAction::AddObstacle);
            button(panel, "Add Climb Box", UiAction::AddClimbStep);
            button(panel, "Change Destination", UiAction::ChangeGoal);
            button(panel, "Clear Obstacles", UiAction::ClearObstacles);

            panel.spawn((
                Text::new("Ready."),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.05, 0.38, 0.68)),
                PlacementText,
            ));

            section(panel, "LIVE");
            panel.spawn((
                Text::new("Starting controller..."),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.08, 0.11, 0.16)),
                TelemetryText,
            ));

            section(panel, "MOUSE");
            muted(
                panel,
                "Left drag: move obstacle\nRight drag: orbit\nMiddle drag: pan\nWheel: zoom",
            );
        });

    // RIGHT PANEL - fewer rows, no overlap.
    commands
        .spawn((
            Node {
                width: px(RIGHT_PANEL),
                height: percent(100),
                position_type: PositionType::Absolute,
                right: px(0),
                top: px(0),
                flex_direction: FlexDirection::Column,
                row_gap: px(7),
                padding: UiRect::all(px(12)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.94, 0.96, 0.98)),
            ZIndex(20),
        ))
        .with_children(|panel| {
            heading(panel, "ROBOT MODEL");

            muted(
                panel,
                "Red cube = Servo A (swing)\nOrange cube = Servo B (lift)\nEach leg has 2 visible servos.",
            );

            section(panel, "LEG GEOMETRY");
            setting_row(panel, "A -> B", SettingKey::ABLength);
            setting_row(panel, "B -> Foot", SettingKey::BFootLength);
            setting_row(panel, "Link width", SettingKey::LinkWidth);

            section(panel, "BODY");
            setting_row(panel, "Body width", SettingKey::BodyWidth);
            setting_row(panel, "Body length", SettingKey::BodyLength);
            setting_row(panel, "Body mass", SettingKey::BodyMass);

            section(panel, "LEG MASS MODEL");
            setting_row(panel, "A-B mass", SettingKey::ABMass);
            setting_row(panel, "B-Foot mass", SettingKey::BFootMass);

            section(panel, "TIMING");
            muted(
                panel,
                "Controller: 50 Hz\nControl budget: 20 ms\nGait: 1.0 s / leg\nFull crawl: 4.0 s",
            );

            section(panel, "MAP LEGEND");
            muted(
                panel,
                "White arrow: robot front\nYellow arrow: command\nGreen line: destination\nBrown box: climb experiment",
            );
        });
}

fn heading(parent: &mut ChildSpawnerCommands, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        TextColor(Color::srgb(0.04, 0.12, 0.22)),
    ));
}

fn section(parent: &mut ChildSpawnerCommands, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: FontSize::Px(11.5),
            ..default()
        },
        TextColor(Color::srgb(0.05, 0.42, 0.78)),
        Node {
            margin: UiRect::top(px(4)),
            ..default()
        },
    ));
}

fn muted(parent: &mut ChildSpawnerCommands, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: FontSize::Px(11.5),
            ..default()
        },
        TextColor(Color::srgb(0.30, 0.35, 0.42)),
    ));
}

fn button(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    action: UiAction,
) {
    parent
        .spawn((
            Button,
            Node {
                width: percent(100),
                height: px(30),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.16, 0.43, 0.75)),
            action,
        ))
        .with_child((
            Text::new(label),
            TextFont {
                font_size: FontSize::Px(12.5),
                ..default()
            },
            TextColor(Color::WHITE),
        ));
}

fn setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    key: SettingKey,
) {
    parent
        .spawn(Node {
            width: percent(100),
            height: px(30),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(4),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont {
                    font_size: FontSize::Px(11.5),
                    ..default()
                },
                TextColor(Color::srgb(0.12, 0.16, 0.22)),
                Node {
                    width: px(105),
                    ..default()
                },
            ));

            tiny_button(
                row,
                "-",
                UiAction::Adjust(key, -1.0),
            );

            row.spawn((
                Text::new("--"),
                TextFont {
                    font_size: FontSize::Px(11.5),
                    ..default()
                },
                TextColor(Color::srgb(0.05, 0.10, 0.16)),
                Node {
                    width: px(82),
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                SettingText(key),
            ));

            tiny_button(
                row,
                "+",
                UiAction::Adjust(key, 1.0),
            );
        });
}

fn tiny_button(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    action: UiAction,
) {
    parent
        .spawn((
            Button,
            Node {
                width: px(28),
                height: px(26),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.78, 0.84, 0.92)),
            action,
        ))
        .with_child((
            Text::new(label),
            TextFont {
                font_size: FontSize::Px(14.0),
                ..default()
            },
            TextColor(Color::srgb(0.04, 0.10, 0.18)),
        ));
}

// ============================================================================
// UI ACTIONS
// ============================================================================

fn ui_buttons(
    mut buttons: Query<
        (&Interaction, &UiAction),
        (Changed<Interaction>, With<Button>),
    >,
    mut control: ResMut<ControlState>,
    mut placement: ResMut<PlacementState>,
    mut camera: ResMut<CameraRig>,
    mut pose: ResMut<RobotPose>,
    mut clock: ResMut<ControlClock>,
    mut config: ResMut<RobotConfig>,
    obstacles: Query<Entity, With<Obstacle>>,
    mut commands: Commands,
) {
    for (interaction, action) in buttons.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }

        match *action {
            UiAction::Manual => {
                set_manual(&mut control, 0.0, 0.0);
            }

            UiAction::Automatic => {
                control.mode = ControlMode::Automatic;
            }

            UiAction::Forward => {
                set_manual(&mut control, 1.0, 0.0);
            }

            UiAction::Backward => {
                set_manual(&mut control, -0.65, 0.0);
            }

            UiAction::Left => {
                set_manual(&mut control, 0.0, -1.0);
            }

            UiAction::Right => {
                set_manual(&mut control, 0.0, 1.0);
            }

            UiAction::Stop => {
                set_manual(&mut control, 0.0, 0.0);
            }

            UiAction::ResetRobot => {
                reset_robot_pose(&mut pose);
                clock.reset_pending = true;
                set_manual(&mut control, 0.0, 0.0);
            }

            UiAction::ResetCamera => {
                *camera = CameraRig::default();
            }

            UiAction::AddObstacle => {
                placement.mode = PlacementMode::Obstacle;
            }

            UiAction::AddClimbStep => {
                placement.mode = PlacementMode::ClimbStep;
            }

            UiAction::ChangeGoal => {
                placement.mode = PlacementMode::Goal;
            }

            UiAction::ClearObstacles => {
                for e in obstacles.iter() {
                    commands.entity(e).despawn();
                }
            }

            UiAction::Adjust(key, direction) => {
                adjust_config(&mut config, key, direction);
            }
        }
    }
}

fn adjust_config(
    config: &mut RobotConfig,
    key: SettingKey,
    direction: f32,
) {
    let s = direction.signum();

    match key {
        SettingKey::ABLength => {
            config.links.coxa_length_m =
                (config.links.coxa_length_m + s * 0.005)
                    .clamp(0.035, 0.120);
        }

        SettingKey::BFootLength => {
            config.links.femur_length_m =
                (config.links.femur_length_m + s * 0.005)
                    .clamp(0.060, 0.180);
        }

        SettingKey::LinkWidth => {
            config.links.link_radius_m =
                (config.links.link_radius_m + s * 0.001)
                    .clamp(0.007, 0.025);
        }

        SettingKey::BodyWidth => {
            config.chassis.width_m =
                (config.chassis.width_m + s * 0.010)
                    .clamp(0.10, 0.28);
        }

        SettingKey::BodyLength => {
            config.chassis.length_m =
                (config.chassis.length_m + s * 0.010)
                    .clamp(0.14, 0.36);
        }

        SettingKey::BodyMass => {
            config.chassis.mass_kg =
                (config.chassis.mass_kg + s * 0.10)
                    .clamp(0.20, 3.0);
        }

        SettingKey::ABMass => {
            config.links.coxa_mass_kg =
                (config.links.coxa_mass_kg + s * 0.010)
                    .clamp(0.02, 0.30);
        }

        SettingKey::BFootMass => {
            config.links.femur_mass_kg =
                (config.links.femur_mass_kg + s * 0.010)
                    .clamp(0.02, 0.40);
            config.links.tibia_mass_kg =
                (config.links.tibia_mass_kg + s * 0.010)
                    .clamp(0.02, 0.40);
        }
    }
}

// ============================================================================
// UI DISPLAY
// ============================================================================

fn update_ui(
    config: Res<RobotConfig>,
    control: Res<ControlState>,
    action: Res<ActionState>,
    pose: Res<RobotPose>,
    world: Res<WorldState>,
    placement: Res<PlacementState>,
    mut telemetry_q: Query<
        &mut Text,
        (
            With<TelemetryText>,
            Without<PlacementText>,
            Without<SettingText>,
        ),
    >,
    mut placement_q: Query<
        &mut Text,
        (
            With<PlacementText>,
            Without<TelemetryText>,
            Without<SettingText>,
        ),
    >,
    mut setting_q: Query<
        (&SettingText, &mut Text),
        (
            Without<TelemetryText>,
            Without<PlacementText>,
        ),
    >,
) {
    let distance = Vec2::new(
        world.goal.x - pose.position.x,
        world.goal.z - pose.position.z,
    )
    .length();

    let eta = if pose.measured_speed > 0.01 {
        format!("{:.1} s", distance / pose.measured_speed)
    } else {
        "--".to_string()
    };

    let mode = match control.mode {
        ControlMode::Manual => "MANUAL",
        ControlMode::Automatic => "AUTOMATIC",
    };

    let command = if action.forward > 0.10 && action.turn.abs() < 0.20 {
        "FORWARD"
    } else if action.forward < -0.10 && action.turn.abs() < 0.30 {
        "BACKWARD"
    } else if action.turn < -0.20 {
        "LEFT"
    } else if action.turn > 0.20 {
        "RIGHT"
    } else {
        "STOP"
    };

    for mut text in telemetry_q.iter_mut() {
        text.0 = format!(
            "Mode: {mode}\n\
             Command: {command}\n\
             Speed: {:.2} cm/s\n\
             Goal: {:.2} m away\n\
             ETA: {eta}\n\
             Active leg: {}\n\
             Phase: {}\n\
             Climb mode: {}",
            pose.measured_speed * 100.0,
            distance,
            action.active_leg,
            action.gait_phase,
            if action.climb_mode { "YES" } else { "no" },
        );
    }

    let placement_message = match placement.mode {
        PlacementMode::None => "Ready.",
        PlacementMode::Obstacle => "Click map to place obstacle.",
        PlacementMode::ClimbStep => "Click map to place climb box.",
        PlacementMode::Goal => "Click map to set destination.",
    };

    for mut text in placement_q.iter_mut() {
        text.0 = placement_message.to_string();
    }

    for (marker, mut text) in setting_q.iter_mut() {
        text.0 = setting_value(&config, marker.0);
    }
}

fn setting_value(
    config: &RobotConfig,
    key: SettingKey,
) -> String {
    match key {
        SettingKey::ABLength => {
            format!("{:.0} mm", config.links.coxa_length_m * 1000.0)
        }
        SettingKey::BFootLength => {
            format!("{:.0} mm", config.links.femur_length_m * 1000.0)
        }
        SettingKey::LinkWidth => {
            format!("{:.0} mm", config.links.link_radius_m * 2000.0)
        }
        SettingKey::BodyWidth => {
            format!("{:.0} mm", config.chassis.width_m * 1000.0)
        }
        SettingKey::BodyLength => {
            format!("{:.0} mm", config.chassis.length_m * 1000.0)
        }
        SettingKey::BodyMass => {
            format!("{:.2} kg", config.chassis.mass_kg)
        }
        SettingKey::ABMass => {
            format!("{:.3} kg", config.links.coxa_mass_kg)
        }
        SettingKey::BFootMass => {
            format!(
                "{:.3} kg",
                config.links.femur_mass_kg + config.links.tibia_mass_kg,
            )
        }
    }
}

fn update_button_colors(
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<Button>),
    >,
) {
    for (interaction, mut color) in buttons.iter_mut() {
        *color = match *interaction {
            Interaction::Pressed => {
                BackgroundColor(Color::srgb(0.04, 0.30, 0.60))
            }
            Interaction::Hovered => {
                BackgroundColor(Color::srgb(0.24, 0.52, 0.84))
            }
            Interaction::None => {
                BackgroundColor(Color::srgb(0.16, 0.43, 0.75))
            }
        };
    }
}

// ============================================================================
// DEBUG / GRID
// ============================================================================

fn draw_world_debug(
    mut gizmos: Gizmos,
    pose: Res<RobotPose>,
    action: Res<ActionState>,
    world: Res<WorldState>,
    drag: Res<DragState>,
    obstacles: Query<(&Transform, &Obstacle)>,
) {
    // 25 cm grid.
    for i in -10..=10 {
        let p = i as f32 * 0.25;

        let color = if i % 4 == 0 {
            Color::srgba(0.20, 0.28, 0.35, 0.55)
        } else {
            Color::srgba(0.25, 0.31, 0.37, 0.28)
        };

        gizmos.line(
            Vec3::new(-2.5, 0.004, p),
            Vec3::new(2.5, 0.004, p),
            color.clone(),
        );

        gizmos.line(
            Vec3::new(p, 0.004, -2.5),
            Vec3::new(p, 0.004, 2.5),
            color,
        );
    }

    let origin = pose.position + Vec3::Y * 0.12;

    let forward = Vec3::new(
        pose.heading.sin(),
        0.0,
        pose.heading.cos(),
    );

    let right = Vec3::new(
        pose.heading.cos(),
        0.0,
        -pose.heading.sin(),
    );

    // White = actual facing direction.
    gizmos.arrow(
        origin,
        origin + forward * 0.26,
        Color::WHITE,
    );

    // Yellow = movement command.
    let command_vector =
        forward * action.forward + right * action.turn;

    if command_vector.length_squared() > 0.001 {
        gizmos.arrow(
            origin + Vec3::Y * 0.04,
            origin
                + Vec3::Y * 0.04
                + command_vector.normalize() * 0.34,
            Color::srgb(1.0, 0.78, 0.06),
        );
    }

    // Green line and beacon = destination.
    gizmos.line(
        Vec3::new(pose.position.x, 0.015, pose.position.z),
        Vec3::new(world.goal.x, 0.015, world.goal.z),
        Color::srgb(0.08, 0.85, 0.20),
    );

    gizmos.circle(
        Isometry3d::new(
            Vec3::new(world.goal.x, 0.016, world.goal.z),
            Quat::from_rotation_x(
                -std::f32::consts::FRAC_PI_2,
            ),
        ),
        0.13,
        Color::srgb(0.08, 0.95, 0.22),
    );

    gizmos.line(
        Vec3::new(world.goal.x, 0.02, world.goal.z),
        Vec3::new(world.goal.x, 0.36, world.goal.z),
        Color::srgb(0.08, 0.95, 0.22),
    );

    if let Some(selected) = drag.obstacle {
        if let Ok((tf, obstacle)) = obstacles.get(selected) {
            gizmos.circle(
                Isometry3d::new(
                    Vec3::new(
                        tf.translation.x,
                        0.012,
                        tf.translation.z,
                    ),
                    Quat::from_rotation_x(
                        -std::f32::consts::FRAC_PI_2,
                    ),
                ),
                obstacle.radius + 0.04,
                Color::srgb(1.0, 0.85, 0.10),
            );
        }
    }
}