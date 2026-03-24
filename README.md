# ICDL_From-Language-to-Drives

## Project overview

A research-grade multimodal pipeline for human-robot alignment: from a natural-language-purpose and vision/input data, to mission and drive generation, and configurable simulation/experimental results.

This repo integrates:
- LLM-based purpose alignment (`Alignment_LLM`).
- Mission extraction (`Missions_LLM`) with weighted mission tagging.
- Drive generation (`Drives_LLM`) with rigorous mathematical function constraints.
- A GUI/engine prototype (`Alignment_and_Translation_Engine/Multimodal_test.py`).

---

## Directory structure and file details

```
ICDL_From-Language-to-Drives/
  Alignment_and_Translation_Engine/
    Multimodal_test.py
  Alignment_LLM/
    Alignment_Engine_LLM.yaml
    Robot_Perceptions.yaml
    VLM_prompt.yaml
  Drives_LLM/
    drives_prompt.yaml
    Robot_Perceptions.yaml
  Missions_LLM/
    internal_needs.yaml
    missions_prompt.yaml
    Robot_Perceptions.yaml
  Results_Alignment_and_Translation_Engine/
    Results_Ideal_Purpose.xlsx
    Results_Multimodal.xlsx
    Results_Unimodal.xlsx
  Results_Gazebo_Simulation/
  Tested_Purposes/
    Purposes
  Videos/
    Gazebo_SImulation/OSCAR_simulation_experiment.mp4
    Real_Robot/Links/
```


## Alignment_and_Translation_Engine

### `Multimodal_test.py`
- Tkinter GUI + scrolled text chat UI for human-robot interaction.
- Supports: save/load YAML dialog, text conversation, voice command record/transcribe (via `whisper`, `sounddevice`, `gtts`, `playsound`).
- Uses `ollama` and `OpenAI/OpenRouter` backends via wrapper `LLMmodel`.
- Provides image-to-base64 encoding for VLM reasoning.
- `load_configuration(config_file)` reads YAML, returns `model` and `initial_prompt`.
- `extract_missions_and_drives(text)` parses LLM output on `<MissionN:[tag,value]> Drive: ...` with regex.
- Chat modes: alignment, mission, drives, VLM sampling, and internal needs/perceptions incorporation.

Use case:
1. Load prompt YAMLs in initial model clients.
2. Traverse user conversation and quantize final purpose.
3. Map purpose -> missions -> drives and optionally visualize.


## Alignment_LLM

### `Alignment_Engine_LLM.yaml`
- `model: phi4:14b` (local/enterprise model endpoint expected). 
- System prompt: Complex pipeline request:
  - Purpose interpretation and dialog refinement.
  - Mission definition: internal, domain independent.
  - Drive definition: numeric urgency.
  - Perceptions and constraints for robotic arm (table coordinates normalized [0,1]).
  - Output format: final description only in `<Final message>`.
- Enforces no high-level generic steps.
- Includes a setup and context block for robot world semantics.

### `Robot_Perceptions.yaml` (Alignment_LLM)
- `object1` (button) and `object2/object3` (apples) with labeled positions, diameter, color, state.
- Introduces `robot_hand` and object perception semantics.

### `VLM_prompt.yaml`
- `model: gpt-4o`.
- Vision-language prompt requiring exactly two sections:
  1. Scene description with matching object list.
  2. Inferred human intent.


## Drives_LLM

### `drives_prompt.yaml`
- `model: gpt-4.1`.
- System prompt for generating drive functions from missions.
- Strict definitions for drive functions and mathematical constraints:
  - Non-negative, 0 only when mission satisfied.
  - Continuous, monotonic, no product/division/singularities.
  - Robots states and object perceptions as variable names.
- Includes mandate: no binary (0/1), no comparisons, no `abs`, no multi-zero conditions.

### `Robot_Perceptions.yaml` (Drives_LLM)
- Identical object configuration as Alignment_LLM.


## Missions_LLM

### `missions_prompt.yaml`
- `model: phi4:14b`.
- Maps human purpose to mission tags and weights `[0.4, 0.8]`: ordering by urgency.
- Explicit: mission tags (no spaces), mission end-state semantics, not actions/verbs.
- Needs weights must be exceeded by mission weights.
- Output: strict `[[tag,weight], Mission2,...]` in `<Answer>` format.

### `internal_needs.yaml`
- 4 cognitive needs defined with weights:
  - `novelty_need` 0.1
  - `effectance_need` 0.25
  - `external_effects_need` 0.3
  - `prospection_need` 0.8

### `Robot_Perceptions.yaml` (Missions_LLM)
- Same object setup as other modules for consistency.


## Results_Alignment_and_Translation_Engine

- Contains generated sample result tables in XLSX format:
  - `Results_Ideal_Purpose.xlsx`
  - `Results_Multimodal.xlsx`
  - `Results_Unimodal.xlsx`

## Results_Gazebo_Simulation

- Simulation file results.

## Tested_Purposes

- Contains a list of all the experimental vague purposes used to validate the Alignment Engine.

## Videos

- `Gazebo_SImulation/OSCAR_simulation_experiment.mp4`: demonstration run.
- `Real_Robot/Links/`: URLs for real robot validation.


---

## Usage guide

1. Create Python virtual env and install dependencies:
   - `pip install tkinter pyyaml ollama openai yamlloader sounddevice numpy torch whisper gtts playsound pillow`
2. Populate API keys in `Multimodal_test.py`:
   - `OpenAI(api_key="API-KEY", base_url="https://openrouter.ai/api/v1")`
3. Run the GUI:
   - `python Alignment_and_Translation_Engine/Multimodal_test.py`
4. Follow prompts to select objects, define user purpose, process mission/drive generation.
5. Inspect results in `Results_Alignment_and_Translation_Engine` and replay `Videos/*` for evidence.
