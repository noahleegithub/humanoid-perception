# Capture Protocol

How to record the two phone videos that `humanoid-perception` compares. Read this end-to-end
**before** you pick up the phone — a bad capture cannot be fixed downstream.

---

## 1. Goal and what "success" looks like

You are filming the **same small indoor room twice**: once as it normally is ("before") and once
after you deliberately change it ("after"). The pipeline:

1. reconstructs each clip in 3D (monocular feed-forward VGGT, on a Colab GPU),
2. **registers** the two rooms into one coordinate frame, and
3. does a visibility-aware voxel comparison to find added / removed / moved objects.

Success = two clips that **reconstruct well and overlap**:

| Signal | Good |
| --- | --- |
| Reconstruction | Both rooms come out dense and recognisable — walls, floor, and large furniture are solid, not smeared |
| Overlap | Most of the room (say 60–80% of static structure) is seen by **both** clips, so registration locks on |
| Alignment | The two point clouds sit on top of each other with a correct scale (walls don't slide/double) |
| Changes | Every change you actually made shows up; very few "phantom" changes appear |

If there is little shared static structure, registration fails. If a change is never seen from a
usable angle in one clip, it will not be reported — that behaviour is intentional.

---

## 2. Pre-flight checklist

- [ ] **Lighting is on and steady** — turn on the room lights, open/close blinds *before* you start, then leave them alone.
- [ ] **Exposure / focus locked** — tap-and-hold the viewfinder to lock AE/AF; disable auto-flash.
- [ ] **Lens is clean** — wipe it; smudges create blur and fake texture.
- [ ] **People and pets are out of the room** for both clips.
- [ ] **Note the room** — its size, main furniture, and layout (helps you retrace the path and sanity-check results).
- [ ] **Tripod/mount preferred**, otherwise steady hands. No gimbal switching mid-clip.
- [ ] **Battery + storage** ready; close other apps so no notification stalls the recording.

---

## 3. Recording protocol (do this for EACH clip)

- **Length:** ~20–40 s. Long enough to see everything, short enough to stay smooth.
- **One continuous take.** No cuts, no stopping, no re-framing restarts.
- **Move in a smooth arc or loop around the room.** Walk the space — do not stand still and do not
  rotate in place. Monocular reconstruction needs **translation** (the camera actually changing
  position), not just rotation, to recover geometry and scale.
- **Keep ~60–80% visual overlap** with where you have already been: as you move, the new view should
  still contain much of what was on screen a moment ago. Sweep so **every wall and large object is
  seen** at least once from a few angles.
- **Height:** hold the phone roughly at chest / eye height, **~1.2–1.6 m**. Keep it fairly level.
- **Slow, smooth motion.** No fast pans or whips. Avoid motion blur.
- **Pause briefly** at the very start and very end (hold still ~1 s) so the first/last frames are sharp.
- **Same lighting between clips.** If you change the lights, the shading change can look like a
  geometric change.
- **Avoid direct windows / backlight** — bright windows blow out and confuse geometry.
- **No zoom at all**, and **do not switch to ultrawide mid-clip** — pick a lens and stay there. Keep
  the camera app stable (no filters, no live effects).

### Suggested path for a small room

1. Start at a corner / doorway, pause, then slowly walk along one wall.
2. Sweep across the far wall, then back along the opposite side, forming a broad **U** or **loop**.
3. End near where you started (or on the far side), pause, stop.
4. Keep the phone pointed slightly ahead of where you're walking so you cover new ground while
   retaining overlap with the previous view.

---

## 4. Doing it twice (the change)

1. Follow §3 exactly and record **clip "before"**.
2. Make **2–3 physical changes**, e.g.:
   - move a chair ~1 m to a clearly different spot,
   - place a box on the floor,
   - remove a mug / small plant from a table.
3. Record **clip "after"** retracing a **similar but not identical path** — deliberately walk from a
   slightly different start point / viewpoint, while still **covering the whole room** and the
   changed objects.

Do **not** try to reproduce the exact same camera path frame-for-frame. A somewhat different
trajectory is good (it gives the registration more viewpoints), as long as the static structure
overlaps and nothing is left unviewed.

---

## 5. Good vs bad examples

| Good | Why it works |
| --- | --- |
| Smooth **loop/arc that translates** around the room, covering all walls | Gives parallax + full coverage → geometry, scale, and overlap |
| ~60–80% overlap with the previous view throughout | Enough shared static structure for reliable registration |
| Phone held level at ~1.2–1.6 m, slow steady walk | Sharp frames, correct, stable geometry |
| Both clips lit identically, no flash | No photometric changes masquerading as 3D changes |
| Each changed object seen from several angles in **both** clips | Change is actually observed |

| Bad | Why it fails |
| --- | --- |
| Stand in one corner and rotate 360° in place | Monocular geometry needs **translation**; pure rotation gives weak/ambiguous scale and poor depth |
| Fast panning / blurry frames | Keyframes get rejected (sharpness filter) and tracking degrades |
| Flash on in one clip only | Lighting change is read as a surface change → spurious "changes" |
| Only 20–30% overlap between clips | Registration can't lock; clouds don't line up |
| Zooming / switching ultrawide mid-clip | Breaks consistent intrinsics; geometry becomes inconsistent |
| Leaving a change occluded or out of frame in one clip | It was never observed → cannot (and won't) be reported |
| Walking through / knocking the camera | Jumps break the continuous trajectory |

---

## 6. Do / Don't quick table

| Always Do | Never Do |
| --- | --- |
| One continuous 20–40 s take per clip | Cut, stop, or restart mid-clip |
| Walk a smooth arc/loop (translate) | Stand still and spin in place |
| Keep 60–80% overlap with the last view | Jump to completely unrelated views |
| Light the room once and keep it fixed | Turn lights on/off or use flash between clips |
| Lock exposure/focus; clean the lens | Rely on auto exposure/auto focus |
| Keep ~fixed zoom and the same lens | Zoom or switch to ultrawide mid-clip |
| Pause ~1 s at start and end | Start/stop while already moving fast |
| Make 2–3 clear, sizeable changes | Make tiny (<10 cm) or purely lighting changes |

---

## 7. What is detectable — and what is not

Detection is voxel-based with a resolution of **0.06 m** (`change.voxel_size`) and a noise/registration
band of ~0.16 m (`change.surface_band_m`), so **roughly ≥10–15 cm** objects are the practical target.

| Detectable ✅ | Not detectable ❌ |
| --- | --- |
| Moving furniture (chair, stool) ~1 m | Small items smaller than ~10 cm (keys, phone, pen) |
| Adding/removing objects ≳ 10–15 cm (box, plant, bin) | Purely **photometric** changes (a light turned on, a poster's colour) |
| Adding/removing medium furniture | Anything **never observed** (occluded or out of view in one/both clips) |
| Object moved within ~1.6 m (paired as removed + added) | Sub-voxel nudges of a few cm |
| Changes with a visible presence in **both** clips | Changes hidden behind a door / inside a cupboard |

Rule of thumb: **if a human can clearly see it in both clips from somewhere on the path, and it's
bigger than a fist, expect it to be found.**

---

## 8. Transferring clips to Colab and running the pipeline

1. Produce two files locally, e.g. `before.mp4` and `after.mp4` (MP4/H.264 is safest; MOV works too).
2. Get them onto Colab **either** way:
   - **Google Drive:** upload to a folder, then in Colab mount it
     (`from google.colab import drive; drive.mount('/content/drive')`).
   - **Direct upload:** use the Colab file panel or
     `from google.colab import files; files.upload()`.
3. Run the pipeline, passing the two clips explicitly:

   ```bash
   hp run --video-a before.mp4 --video-b after.mp4
   ```

   Useful options: `--out outputs/room01`, `--config configs/default.yaml`,
   `--set change.voxel_size=0.06`, `--no-semantics` (skip SAM 3.1 labelling). Use `--set` to
   override any config value, e.g. `--set io.max_keyframes=12`, `--set io.long_side=518`.

**Naming:** what matters is order — `--video-a` is the **before** clip, `--video-b` is the **after**
clip. Keep names obvious (`before.mp4` / `after.mp4`) so you don't invert them.

**Capture-side knobs that matter** (from `configs/default.yaml`):

| Knob | Value | Effect on capture |
| --- | --- | --- |
| `io.max_keyframes` | 12 | Only ~12 frames per clip are used → **full coverage per clip is important** |
| `io.long_side` | 518 | Frames are downscaled → very fine texture won't survive; rely on structure |
| `change.voxel_size` | 0.06 m | Voxel size → sets the ~10–15 cm detection floor for object size |

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| "Alignment failed" / clouds don't sit together | Too little shared static structure | Re-record with **more overlap** (60–80%) and more of the same walls/furniture in both clips |
| Scale looks wrong (room too big/small, doubled walls) | Too much in-place rotation, weak translation, or little overlap | Walk/translate more, keep a steady height, cover identical static structure |
| A change you made is **missing** | The object was occluded or never in view in **both** clips | Re-record so that object is clearly seen from a few angles in each pass |
| **Spurious** changes everywhere | Lighting shift between clips, or heavy motion blur | Keep lighting identical (no flash), slow down, pause at start/end, avoid windows/backlight |
| Blurry / few usable frames | Fast motion, auto-focus hunting | Lock focus, walk slower, clean lens |
| Detection seems off even with good clips | Object smaller than ~10 cm, or photometric-only change | Expected limitation (§7) — pick larger, physical changes |

---

*Related:* project overview in `README.md`, pipeline settings in `configs/default.yaml`.
