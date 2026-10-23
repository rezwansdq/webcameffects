"""Neon glow rendering: aura outline (+ optional skeleton) -> trail buffer -> bloom -> composite."""

import cv2
import numpy as np

from .tracking import (
    ARMS, HAND_CONNECTIONS, HAND_TIPS, L_EAR, L_SHOULDER, POSE_BODY_CONNECTIONS, POSE_BODY_JOINTS,
    POSE_HANDS, R_EAR, R_SHOULDER,
)

VISIBILITY_MIN = 0.5
BLOOM_SCALE = 4  # bloom is computed at 1/4 resolution for speed
PALM = [0, 1, 2, 5, 9, 13, 17]


def _pt(p):
    return int(p[0]), int(p[1])


def _disk(r):
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_small_holes(mask, max_frac=0.03):
    """Fill gaps in the person mask (e.g. dark hair against a dark wall) so the
    outline doesn't trace them. Large gaps, like between an arm and the body, stay."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    limit = max_frac * mask.shape[0] * mask.shape[1]
    holes = [c for c, hc in zip(contours, hierarchy[0]) if hc[3] != -1 and cv2.contourArea(c) < limit]
    if holes:
        cv2.drawContours(mask, holes, -1, 255, -1)
    return mask


class GlowRenderer:
    def __init__(self):
        self.trail = None  # uint8 mask that fades over time
        self.prev_tracks = None

    def reset(self):
        self.trail = None
        self.prev_tracks = None

    # ---- aura: glowing outline hugging the hands, arms and head ------------

    def _aura_masks(self, tracks, s, w, h):
        """Return (ring, core) uint8 masks at frame size, or None.

        The person segmentation gives the true silhouette. Landmarks pick which
        parts of it get an outline: hands, arms, head. The outline is drawn
        just outside the silhouette, and only where it borders the background,
        except around hands, which also get outlined in front of the body.
        """
        person_f = tracks.get("person")
        if person_f is None:
            return None
        mh, mw = person_f.shape
        f = mw / w  # frame coords -> mask coords
        person = _fill_small_holes((person_f > 0.5).astype(np.uint8) * 255)

        zone = np.zeros((mh, mw), np.uint8)        # parts outlined against the background
        hand_shape = np.zeros((mh, mw), np.uint8)  # hand silhouettes from landmarks

        pose = tracks.get("pose")
        vis = None
        if pose is not None:
            pose = pose.copy()
            pose[:, :2] *= f
            vis = pose[:, 2] >= VISIBILITY_MIN

        if pose is not None and vis[L_SHOULDER] and vis[R_SHOULDER]:
            ref = np.linalg.norm(pose[L_SHOULDER, :2] - pose[R_SHOULDER, :2])
        else:
            ref = mw * 0.2

        if s.aura_arms and pose is not None:
            arm_w = max(6, int(ref * 0.55))
            for sh, el, wr in ARMS:
                if vis[sh] and vis[el]:
                    cv2.line(zone, _pt(pose[sh]), _pt(pose[el]), 255, arm_w)
                if vis[el] and vis[wr]:
                    cv2.line(zone, _pt(pose[el]), _pt(pose[wr]), 255, arm_w)
                    cv2.circle(zone, _pt(pose[wr]), arm_w // 2, 255, -1)

        if s.aura_head and pose is not None and vis[L_EAR] and vis[R_EAR]:
            r = np.linalg.norm(pose[L_EAR, :2] - pose[R_EAR, :2])
            cx, cy = (pose[L_EAR, :2] + pose[R_EAR, :2]) / 2
            cv2.ellipse(zone, (int(cx), int(cy - 0.35 * r)), (int(1.4 * r), int(1.9 * r)), 0, 0, 360, 255, -1)

        hands = [hnd * f for hnd in tracks.get("hands", [])] if s.aura_hands else []
        for hand in hands:
            size = max(4.0, np.linalg.norm(hand[0] - hand[9]))
            finger_w = max(3, int(size * 0.32))
            cv2.fillConvexPoly(hand_shape, cv2.convexHull(hand[PALM].astype(np.int32)), 255)
            for a, b in HAND_CONNECTIONS:
                cv2.line(hand_shape, _pt(hand[a]), _pt(hand[b]), 255, finger_w)
            for t in HAND_TIPS:
                cv2.circle(hand_shape, _pt(hand[t]), finger_w // 2, 255, -1)
        if s.aura_hands and not hands and pose is not None:
            # Hand tracker lost the hands: fall back to a zone around the pose's hand points.
            for idx in POSE_HANDS:
                if all(vis[i] for i in idx):
                    c = pose[list(idx), :2].mean(axis=0)
                    cv2.circle(zone, _pt(c), int(ref * 0.45), 255, -1)

        part = cv2.bitwise_and(person, cv2.bitwise_or(zone, hand_shape))
        if not part.any():
            return None

        ring_w = max(2, s.thickness * mw / 960.0)
        ring = cv2.subtract(cv2.dilate(part, _disk(ring_w)), part)
        core = cv2.subtract(cv2.dilate(part, _disk(max(1, ring_w / 3))), part)

        # Only outline against the background, plus around hands over the body.
        allowed = cv2.bitwise_not(person)
        if hands:
            allowed = cv2.bitwise_or(allowed, cv2.dilate(hand_shape, _disk(ring_w * 2)))
        ring = cv2.bitwise_and(ring, allowed)
        core = cv2.bitwise_and(core, allowed)

        ring = cv2.resize(ring, (w, h), interpolation=cv2.INTER_LINEAR)
        core = cv2.resize(core, (w, h), interpolation=cv2.INTER_LINEAR)
        return ring, core

    # ---- skeleton: neon lines drawn on the body (optional) -----------------

    def _skeleton_masks(self, shape, tracks, s):
        h, w = shape
        glow = np.zeros((h, w), np.uint8)
        core = np.zeros((h, w), np.uint8)
        scale = w / 960.0
        thick = max(2, int(s.thickness * scale))
        core_thick = max(1, thick // 3)
        dot = max(3, int(thick * 1.3))

        def seg(a, b):
            pa, pb = _pt(a), _pt(b)
            cv2.line(glow, pa, pb, 255, thick, cv2.LINE_AA)
            cv2.line(core, pa, pb, 255, core_thick, cv2.LINE_AA)

        def node(p, r):
            c = _pt(p)
            cv2.circle(glow, c, r, 255, -1, cv2.LINE_AA)
            cv2.circle(core, c, max(1, r // 2), 255, -1, cv2.LINE_AA)

        pose = tracks.get("pose")
        if pose is not None:
            vis = pose[:, 2] >= VISIBILITY_MIN
            for a, b in POSE_BODY_CONNECTIONS:
                if vis[a] and vis[b]:
                    seg(pose[a, :2], pose[b, :2])
            for j in POSE_BODY_JOINTS:
                if vis[j]:
                    node(pose[j, :2], dot)
        for hand in tracks.get("hands", []):
            for a, b in HAND_CONNECTIONS:
                seg(hand[a], hand[b])
            for t in HAND_TIPS:
                node(hand[t], max(2, dot - 1))
        return glow, core

    def _motion_fill(self, glow, tracks, s):
        """Draw in-between skeletons from the previous frame so fast moves leave a
        continuous smear instead of separate stripes at low frame rates."""
        prev = self.prev_tracks
        if prev is None:
            return
        mids = []
        if tracks.get("pose") is not None and prev.get("pose") is not None:
            mids.append(("pose", prev["pose"], tracks["pose"]))
        for hand in tracks.get("hands", []):
            # Match each hand to the previous hand whose wrist was nearest.
            best = min(prev.get("hands", []), key=lambda p: np.linalg.norm(p[0] - hand[0]), default=None)
            if best is not None:
                mids.append(("hand", best, hand))
        if not mids:
            return
        # More steps for bigger jumps, capped to keep it cheap.
        jump = max(float(np.abs(b[:, :2] - a[:, :2]).max()) for _, a, b in mids)
        steps = int(min(6, jump // (max(2.0, s.thickness) * 1.5)))
        for i in range(1, steps + 1):
            t = i / (steps + 1)
            mid = {"pose": None, "hands": []}
            for kind, a, b in mids:
                pts = a + (b - a) * t
                if kind == "pose":
                    mid["pose"] = pts
                else:
                    mid["hands"].append(pts)
            g, _ = self._skeleton_masks(glow.shape, mid, s)
            cv2.max(glow, g, dst=glow)

    # ---- compositing -------------------------------------------------------

    def render(self, frame, tracks, s, effects_on=True):
        """frame: BGR uint8. Returns BGR uint8 output frame."""
        h, w = frame.shape[:2]

        if not effects_on:
            self.reset()
            return frame

        # Darkened, slightly desaturated background so the glow pops.
        gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        dark = 1.0 - s.darkness
        bg = cv2.addWeighted(frame, 0.45 * dark, gray, 0.55 * dark, 0)

        glow = np.zeros((h, w), np.uint8)
        core = np.zeros((h, w), np.uint8)
        if s.skeleton:
            glow, core = self._skeleton_masks((h, w), tracks, s)
            if s.trails:
                self._motion_fill(glow, tracks, s)
        aura = self._aura_masks(tracks, s, w, h)
        if aura is not None:
            cv2.max(glow, aura[0], dst=glow)
            cv2.max(core, aura[1], dst=core)

        # Motion trails: fade the old mask (with a slight smear) and keep the brighter of old/new.
        if s.trails:
            if self.trail is None or self.trail.shape != glow.shape:
                self.trail = glow.copy()
            else:
                decay = 0.55 + 0.42 * s.trail_length  # 0..1 slider -> 0.55..0.97
                faded = cv2.convertScaleAbs(cv2.blur(self.trail, (3, 3)), alpha=decay)
                self.trail = cv2.max(faded, glow)
            mask = self.trail
        else:
            self.trail = None
            mask = glow
        self.prev_tracks = tracks

        # Bloom: blur at low resolution at two radii, upsample, stack.
        sw, sh = max(1, w // BLOOM_SCALE), max(1, h // BLOOM_SCALE)
        small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
        r1 = max(1.0, sw / 160.0)
        b1 = cv2.GaussianBlur(small, (0, 0), r1)
        b2 = cv2.GaussianBlur(small, (0, 0), r1 * 4)
        bloom = cv2.addWeighted(b1, 1.2, b2, 1.8, 0)  # saturating uint8
        bloom = cv2.resize(bloom, (w, h), interpolation=cv2.INTER_LINEAR)

        intensity = s.intensity * 2.0  # 0..1 slider -> 0..2
        energy = cv2.addWeighted(mask, 0.85 * intensity, bloom, intensity, 0)
        hot_gain = min(1.0, intensity)

        # Colorize per channel (all saturating uint8 math, fast on CPU).
        out_ch = []
        for c, bg_ch in zip(s.color_bgr, cv2.split(bg)):
            glow_ch = cv2.convertScaleAbs(energy, alpha=c / 255.0)
            # White-hot core line at the edge, tinted slightly by the color.
            hot_ch = cv2.convertScaleAbs(core, alpha=hot_gain * (0.35 * c + 0.65 * 255) / 255.0)
            out_ch.append(cv2.add(cv2.add(bg_ch, glow_ch), hot_ch))
        return cv2.merge(out_ch)
