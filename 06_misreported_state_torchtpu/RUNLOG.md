# Run log: module 06 student track

A template. Fill it in while you go through the nine steps in [README.md](README.md), one block per step. It is how feedback on the lab gets back to its author. A step that failed, a line that did not match what the README said, a wait nobody warned you about: those are the entries that matter most, so write them down when they happen.

## How to fill it in

- **Prediction:** your answer to the step's Predict question. Write it before you run the command, and leave it as you wrote it.
- **Command run:** `as written`, or the command exactly as you ran it if you had to change it.
- **Pass / fail:** `pass`, `fail`, or `pass with changes`.
- **Output:** paste what the command printed. Trim long output down to the lines the README's Observed or Expect names, and keep anything that surprised you. Paste errors whole.
- **Time taken:** hands-on minutes and waiting minutes, separately.
- **Checkpoint answer:** one or two sentences.
- **Notes:** anything unclear, wrong or missing in the README for this step.

Two cautions. Never paste a Hugging Face token or anything from `gcloud auth` into this file. And fill it in on your own machine, never in the VM's checkout: an edited file there would stamp every run log on the VM as `-dirty`.

## Run header

| Field | Value |
| --- | --- |
| Name | |
| Date | |
| Commit on your machine (`git rev-parse HEAD`) | |
| Commit on the VM (`git rev-parse HEAD`) | |
| Machine, OS, Python and `uv` versions | |
| `gcloud` SDK version | |
| Zone | |
| Hourly price you looked up, with its source and date | |
| VM `RUNNING` at | |
| VM deleted at | |
| Hours the VM existed | |
| Cost from the README's formula | |
| Total hands-on minutes | |
| Total waiting minutes | |

## Step 1: setup and laptop rehearsal

- **Where:** laptop
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- Tests run and result (`Ran N tests`, `OK`):
- Rehearsal's last line:

## Step 2: meet the eight records and the rules baseline

- **Where:** laptop
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- Rules line (`Evaluated 8 records: ...`):
- Record the rules flagged, and the step:

## Step 3: adjudicate the eight yourself

- **Where:** laptop
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- Decisions, one per record (accept, clean, edit, reject), 001 to 008:
- Proposals you changed, and why:
- The `Labels:` line `make misreport-student-set` printed:

## Step 4: create the VM, install, smoke check

- **Where:** laptop, then VM
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- Minutes from the create command to `RUNNING`:
- The `vllm` line from `uv pip list`:
- Bands proposed by `bands.py`, and their ceilings:
- Prompts planned by the dry run:
- The refusal the third smoke command printed, and its exit code:

## Step 5: run the judge on TorchTPU

- **Where:** VM
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- `--dtype` used:
- Seconds per band, from the `pass 1/1` lines:
- Minutes from the command to `Results written`:
- `parse_errors`:
- If the load failed, the error, pasted in full under Output:

## Step 6: compare three readers of the same records

- **Where:** VM
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- The eight-row table, pasted under Output:
- The counts line under it:
- A record where the three readers disagree, and which one you believe:

## Step 7: the same eval set on the JAX backend, then parity

- **Where:** VM
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- Did JAX load the 26B checkpoint (yes or no)? If no, the error, pasted in full under Output:
- Fallback branch taken (yes or no):
- Minutes from the docker command to `Results written`:
- Decisions that agree, of 132, and the `disagree_ids`:
- Anything parity refused, and the field it named:

## Step 8: bridge to the rai-checklist-cli report

- **Where:** VM
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- Sections that read FAIL:
- The `label_sources` line from the provenance block:

## Step 9: tear down the VM and confirm nothing is running

- **Where:** VM, then laptop
- **Prediction:**
- **Command run:** as written
- **Pass / fail:**
- **Output:**

```text

```

- **Time taken:** hands-on ___ min, waiting ___ min
- **Checkpoint answer:**
- **Notes:**

From this step:

- Files that came back, from `ls results/student-vm/student`:
- Output of the three list commands after the delete, pasted under Output:
- Time the VM was deleted:

## Extension exercises

Optional. One line each if you did them.

- **1. The determinism control:** decisions TorchTPU gave the same way twice, of 132:
- **2. Break the sampling setting:** the two tests that failed:
- **3. Draft a ninth record:** what it covers, and the provenance sentence you wrote for it:

## Where the README and the run disagreed

One row for every Observed or Expect line that did not match what you saw.

| Step | The README says | I saw |
| --- | --- | --- |
| | | |

## Findings

Anything that is a result in its own right: a checkpoint that would not load on a backend, a parse-error count above zero, a refusal you did not expect. Name the file under `results/student/` that holds the evidence.

- 

## Three things to change in the lab

1. 
2. 
3. 
