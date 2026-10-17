bl_info = {
    "name": "SubWater - Subtractive Watercolor Texture Paint",
    "author": "Custom 3D Pipeline",
    "version": (2, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N-Panel) > SubWater (Texture Paint Mode)",
    "description": "Real-time subtractive watercolor brush with Luminance Alpha Texture channel, 10-100ms GPU sync, and UI passthrough.",
    "category": "Paint",
}

import bpy
import bmesh
import math
import time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader


# --- STATE & LUMINANCE CACHE ---

class SubWaterRuntimeState:
    active = False
    cached_tex_name = ""
    cached_tex_luma = None


def get_cached_luminance_map(img):
    """Extracts and caches a 2D float32 luminance array (H, W) from a Blender Image."""
    if img is None:
        return None
    w, h = img.size[0], img.size[1]
    if w == 0 or h == 0:
        return None

    if (SubWaterRuntimeState.cached_tex_name != img.name or
            SubWaterRuntimeState.cached_tex_luma is None or
            SubWaterRuntimeState.cached_tex_luma.shape != (h, w)):
        raw = np.zeros((h * w * 4,), dtype=np.float32)
        img.pixels.foreach_get(raw)
        rgba = raw.reshape((h, w, 4))
        # Rec.709 Luminance weighted by image alpha
        luma = (0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]) * rgba[:, :, 3]
        SubWaterRuntimeState.cached_tex_luma = np.clip(luma, 0.0, 1.0)
        SubWaterRuntimeState.cached_tex_name = img.name

    return SubWaterRuntimeState.cached_tex_luma


# --- PROPERTIES ---

class SubWaterSettings(bpy.types.PropertyGroup):
    update_interval_ms: bpy.props.IntProperty(
        name="Update Interval (ms)",
        default=25, min=10, max=100,
        description="Real-time viewport texture refresh interval in milliseconds"
    )
    strength: bpy.props.FloatProperty(
        name="Pigment Flow",
        default=0.35, min=0.01, max=1.0,
        description="Pigment concentration per dab"
    )
    water_dilution: bpy.props.FloatProperty(
        name="Water Dilution",
        default=0.40, min=0.0, max=0.95,
        description="Thins pigment and gently washes underlying paint toward paper white"
    )
    edge_darkening: bpy.props.FloatProperty(
        name="Edge Ring (Coffee-Ring)",
        default=0.85, min=0.0, max=2.5,
        description="Concentrates darker pigment along the outer edge of the wet stroke"
    )
    wet_bleed: bpy.props.FloatProperty(
        name="Wet-on-Wet Bleed",
        default=0.45, min=0.0, max=1.0,
        description="Capillary diffusion that bleeds pigments into surrounding pixels"
    )
    uv_radius: bpy.props.IntProperty(
        name="Brush Size (px)",
        default=45, min=2, max=300,
        description="Brush radius in texture pixels"
    )

    # --- LUMINANCE TEXTURE CHANNEL PROPERTIES ---
    use_brush_texture: bpy.props.BoolProperty(
        name="Enable Luminance Alpha Texture",
        default=True,
        description="Use an image's luminance channel as an alpha mask for the brush"
    )
    brush_texture: bpy.props.PointerProperty(
        name="Luminance Map",
        type=bpy.types.Image,
        description="Image whose luminance (brightness) acts as the brush alpha mask"
    )
    tex_mapping: bpy.props.EnumProperty(
        name="Mapping",
        items=[
            ('STAMP', "Brush Tip Stamp", "Maps the luminance texture inside the brush tip circle"),
            ('TILED', "Canvas Paper Grain (Tiled)", "Locks the luminance texture to UV space like cold-press paper grain"),
        ],
        default='STAMP'
    )
    tex_strength: bpy.props.FloatProperty(
        name="Texture Alpha Influence",
        default=0.85, min=0.0, max=1.0,
        description="How strongly the luminance map masks the brush alpha (0 = no mask, 1 = full luminance alpha)"
    )
    tex_scale: bpy.props.FloatProperty(
        name="Grain Tile Scale",
        default=8.0, min=0.5, max=64.0,
        description="Tiling frequency when using Canvas Paper Grain mode"
    )
    tex_contrast: bpy.props.FloatProperty(
        name="Luminance Contrast",
        default=1.5, min=0.2, max=5.0,
        description="Boosts contrast of the luminance alpha map for crisper bristle/grain texture"
    )
    tex_jitter_angle: bpy.props.BoolProperty(
        name="Random Angle Jitter",
        default=True,
        description="Randomly rotates the brush stamp texture on each dab for organic strokes"
    )
    tex_invert: bpy.props.BoolProperty(
        name="Invert Luminance",
        default=False,
        description="Inverts the luminance map (dark pixels paint, bright pixels mask)"
    )

    canvas_size: bpy.props.EnumProperty(
        name="Canvas Resolution",
        items=[
            ('1024', "1024 x 1024 (Fastest)", ""),
            ('2048', "2048 x 2048 (Balanced)", ""),
            ('4096', "4096 x 4096 (High Detail)", ""),
        ],
        default='2048'
    )


# --- AUTOMATIC CANVAS & MATERIAL SETUP ---

def ensure_texture_paint_canvas(context, obj, force_reset=False):
    mesh = obj.data
    res = int(context.scene.subwater_settings.canvas_size)

    if not mesh.uv_layers:
        prev_mode = context.mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02)
        if prev_mode == 'PAINT_TEXTURE':
            bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')

    img_name = f"{obj.name}_SubWater_Canvas"
    img = bpy.data.images.get(img_name)
    if not img or force_reset or img.size[0] != res:
        if img and img.size[0] != res:
            bpy.data.images.remove(img)
            img = None
        if not img:
            img = bpy.data.images.new(
                name=img_name, width=res, height=res,
                alpha=False, float_buffer=False
            )
        paper = np.full((res * res, 4), (0.98, 0.96, 0.92, 1.0), dtype=np.float32)
        img.pixels.foreach_set(paper.ravel())
        img.update()
        img.update_tag()

    mat = obj.active_material
    if not mat:
        mat = bpy.data.materials.new(name=f"{obj.name}_SubWater_Mat")
        obj.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.85

    tex_node = next((n for n in nodes if n.type == 'TEX_IMAGE' and n.image == img), None)
    if not tex_node:
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = img
        tex_node.location = (bsdf.location.x - 320, bsdf.location.y)
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

    for n in nodes:
        n.select = False
    tex_node.select = True
    nodes.active = tex_node

    if context.scene.tool_settings.image_paint:
        ipaint = context.scene.tool_settings.image_paint
        ipaint.mode = 'IMAGE'
        ipaint.canvas = img

    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    if space.shading.type == 'SOLID':
                        space.shading.color_type = 'TEXTURE'

    return img


# --- SUBTRACTIVE WATERCOLOR + LUMINANCE ALPHA ENGINE ---

def sample_luminance_alpha(luma_map, xx, yy, cx, cy, r_px, H, W, settings, dab_angle):
    """Samples the 2D luminance map and returns a normalized alpha mask [0..1] for the dab patch."""
    th, tw = luma_map.shape

    if settings.tex_mapping == 'STAMP':
        # Local brush coordinates normalized to [-1, 1]
        dx = (xx - cx) / max(float(r_px), 1.0)
        dy = (yy - cy) / max(float(r_px), 1.0)
        if settings.tex_jitter_angle:
            ca = math.cos(dab_angle)
            sa = math.sin(dab_angle)
            rx = dx * ca - dy * sa
            ry = dx * sa + dy * ca
        else:
            rx, ry = dx, dy

        u_tex = np.clip((rx * 0.5 + 0.5) * (tw - 1), 0, tw - 1).astype(np.int32)
        v_tex = np.clip((ry * 0.5 + 0.5) * (th - 1), 0, th - 1).astype(np.int32)
    else:
        # Tiled Canvas Paper Grain locked to UV pixel space
        scale = settings.tex_scale
        u_tex = (np.floor((xx / float(W)) * tw * scale).astype(np.int32)) % tw
        v_tex = (np.floor((yy / float(H)) * th * scale).astype(np.int32)) % th

    sampled = luma_map[v_tex, u_tex]
    if settings.tex_invert:
        sampled = 1.0 - sampled

    # Apply contrast curve around midpoint 0.5
    if abs(settings.tex_contrast - 1.0) > 0.01:
        sampled = np.clip((sampled - 0.5) * settings.tex_contrast + 0.5, 0.0, 1.0)

    # Blend between 1.0 (unmasked) and the luminance value based on tex_strength
    return (1.0 - settings.tex_strength) + settings.tex_strength * sampled


def apply_subtractive_watercolor_dab(buf, u, v, r_px, brush_rgb, flow, settings, luma_map=None, dab_angle=0.0):
    """Paints into the numpy pixel buffer using Beer-Lambert subtractive mixing + Luminance Alpha."""
    H, W, _ = buf.shape
    cx = int((u % 1.0) * W)
    cy = int((v % 1.0) * H)

    x0 = max(0, cx - r_px)
    x1 = min(W, cx + r_px + 1)
    y0 = max(0, cy - r_px)
    y1 = min(H, cy + r_px + 1)

    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = dist / max(float(r_px), 1.0)
    mask = r_norm <= 1.0
    if not np.any(mask):
        return

    patch = buf[y0:y1, x0:x1, :3]
    eps = 1e-3

    # 1. Convert existing canvas RGB and Brush RGB to Subtractive Absorption Space (-ln(R))
    abs_base = -np.log(np.clip(patch, eps, 1.0))
    b_rgb = np.clip(np.array(brush_rgb[:3], dtype=np.float32), eps, 1.0)
    abs_brush = -np.log(b_rgb)

    # 2. Radial falloff + Luminance Texture Alpha Channel
    falloff = np.where(mask, (1.0 - r_norm ** 2) ** 1.25, 0.0)

    if settings.use_brush_texture and luma_map is not None:
        luma_alpha = sample_luminance_alpha(luma_map, xx, yy, cx, cy, r_px, H, W, settings, dab_angle)
        falloff = falloff * luma_alpha

    falloff = falloff[..., np.newaxis]
    ring = np.exp(-((r_norm - 0.80) ** 2) / 0.022)[..., np.newaxis]

    dilution = settings.water_dilution
    edge_ring = settings.edge_darkening
    bleed = settings.wet_bleed

    density = (1.0 - dilution * 0.72) * (1.0 + edge_ring * ring * 1.65)
    weight = np.clip(falloff * flow, 0.0, 1.0)

    # 3. Water Dilution gently lifts underlying pigment toward paper white
    wash_factor = 1.0 - (dilution * weight * 0.28)
    abs_base_washed = abs_base * wash_factor

    # 4. Mix in Subtractive Absorption Space
    abs_mixed = abs_base_washed * (1.0 - weight) + (abs_brush * density) * weight

    # 5. Wet-on-Wet Capillary Bleed
    if bleed > 0.01 and patch.shape[0] > 4 and patch.shape[1] > 4:
        abs_blur = (
            abs_mixed
            + np.roll(abs_mixed, 1, axis=0)
            + np.roll(abs_mixed, -1, axis=0)
            + np.roll(abs_mixed, 1, axis=1)
            + np.roll(abs_mixed, -1, axis=1)
        ) * 0.2
        abs_blur[0, :] = abs_mixed[0, :]
        abs_blur[-1, :] = abs_mixed[-1, :]
        abs_blur[:, 0] = abs_mixed[:, 0]
        abs_blur[:, -1] = abs_mixed[:, -1]

        bleed_mask = np.clip(falloff * bleed * 0.65, 0.0, 1.0)
        abs_mixed = abs_mixed * (1.0 - bleed_mask) + abs_blur * bleed_mask

    # 6. Convert back to RGB reflectance
    buf[y0:y1, x0:x1, :3] = np.clip(np.exp(-abs_mixed), 0.0, 1.0)


# --- PROCEDURAL LUMINANCE TEXTURE GENERATOR ---

class IMAGE_OT_subwater_generate_grain_tex(bpy.types.Operator):
    """Generates a ready-to-use Watercolor Spatter & Grain Luminance Texture"""
    bl_idname = "image.subwater_generate_grain_tex"
    bl_label = "Generate Procedural Grain Texture"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.subwater_settings
        tex_name = "SubWater_Luminance_Grain"
        img = bpy.data.images.get(tex_name)
        res = 512
        if not img:
            img = bpy.data.images.new(name=tex_name, width=res, height=res, alpha=False)

        # Build multi-octave organic bristle/grain luminance pattern in NumPy
        y, x = np.mgrid[0:res, 0:res] / float(res)
        rng = np.random.default_rng(seed=42)
        noise_fine = rng.random((res, res), dtype=np.float32)

        # Smooth multi-frequency cellular waves + organic spatter
        waves = (
            np.sin(x * 38.0 + np.cos(y * 31.0) * 2.5) * 0.25 +
            np.cos(y * 52.0 - np.sin(x * 43.0) * 2.5) * 0.25 + 0.5
        )
        # Radial bristle streaks + grain
        luma = np.clip(waves * 0.55 + noise_fine * 0.45, 0.0, 1.0)
        luma = np.clip((luma - 0.45) * 2.1 + 0.5, 0.0, 1.0).astype(np.float32)

        rgba = np.ones((res, res, 4), dtype=np.float32)
        rgba[:, :, 0] = luma
        rgba[:, :, 1] = luma
        rgba[:, :, 2] = luma

        img.pixels.foreach_set(rgba.ravel())
        img.update()

        s.brush_texture = img
        s.use_brush_texture = True
        SubWaterRuntimeState.cached_tex_name = ""  # Force cache refresh
        self.report({'INFO'}, "Generated 'SubWater_Luminance_Grain' and assigned to Brush Texture!")
        return {'FINISHED'}


# --- INTERVAL-SYNCED MODAL TEXTURE PAINT OPERATOR ---

class PAINT_OT_subwater_texpaint_modal(bpy.types.Operator):
    """Paints subtractively with a Luminance Alpha channel and 10-100ms GPU timer sync"""
    bl_idname = "paint.subwater_texpaint_modal"
    bl_label = "Toggle SubWater Brush"
    bl_options = {'REGISTER', 'UNDO'}

    _handle_draw = None
    _timer = None
    is_painting = False
    buffer_dirty = False
    last_gpu_sync_time = 0.0
    hit_world = None
    hit_normal = None
    last_uv = None
    current_uv = None

    def get_active_brush_color(self, context):
        ts = context.tool_settings
        if ts.image_paint and ts.image_paint.brush:
            return ts.image_paint.brush.color
        if ts.unified_paint_settings:
            return ts.unified_paint_settings.color
        return (0.05, 0.4, 0.9)

    def build_bvh_and_uv_cache(self, context, obj):
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.faces.ensure_lookup_table()

        uv_layer = bm.loops.layers.uv.active
        self.tri_data = []
        for f in bm.faces:
            v0 = f.loops[0].vert.co.copy()
            v1 = f.loops[1].vert.co.copy()
            v2 = f.loops[2].vert.co.copy()
            uv0 = f.loops[0][uv_layer].uv.copy()
            uv1 = f.loops[1][uv_layer].uv.copy()
            uv2 = f.loops[2][uv_layer].uv.copy()
            self.tri_data.append((v0, v1, v2, uv0, uv1, uv2))

        self.bvh = BVHTree.FromBMesh(bm)
        bm.free()

        self.mw = obj.matrix_world.copy()
        self.mw_inv = self.mw.inverted()

    def is_mouse_in_viewport_window(self, context, event):
        if not self.view3d_area:
            return False
        for reg in self.view3d_area.regions:
            if reg.type in {'UI', 'HEADER', 'TOOL_HEADER', 'TOOLS', 'HUD'}:
                if (reg.x <= event.mouse_x <= reg.x + reg.width and
                        reg.y <= event.mouse_y <= reg.y + reg.height):
                    return False
        for reg in self.view3d_area.regions:
            if reg.type == 'WINDOW':
                if (reg.x <= event.mouse_x <= reg.x + reg.width and
                        reg.y <= event.mouse_y <= reg.y + reg.height):
                    self.win_region = reg
                    return True
        return False

    def raycast_uv(self, context, event):
        rv3d = self.view3d_space.region_3d
        coord = (event.mouse_x - self.win_region.x, event.mouse_y - self.win_region.y)

        view_vec = view3d_utils.region_2d_to_vector_3d(self.win_region, rv3d, coord)
        ray_orig = view3d_utils.region_2d_to_origin_3d(self.win_region, rv3d, coord)

        orig_local = self.mw_inv @ ray_orig
        dir_local = (self.mw_inv.to_3x3() @ view_vec).normalized()

        loc_local, norm_local, face_idx, _ = self.bvh.ray_cast(orig_local, dir_local)
        if loc_local is not None and face_idx < len(self.tri_data):
            self.hit_world = self.mw @ loc_local
            self.hit_normal = (self.mw.to_3x3() @ norm_local).normalized()

            v0, v1, v2, uv0, uv1, uv2 = self.tri_data[face_idx]
            uv_3d = barycentric_transform(
                loc_local, v0, v1, v2,
                Vector((uv0.x, uv0.y, 0.0)),
                Vector((uv1.x, uv1.y, 0.0)),
                Vector((uv2.x, uv2.y, 0.0))
            )
            return Vector((uv_3d.x, uv_3d.y))

        self.hit_world = None
        self.hit_normal = None
        return None

    def flush_buffer_to_gpu(self, context, force=False):
        if not self.buffer_dirty:
            return

        now = time.perf_counter()
        interval_sec = context.scene.subwater_settings.update_interval_ms / 1000.0

        if force or (now - self.last_gpu_sync_time) >= interval_sec:
            self.img.pixels.foreach_set(self.pixel_buf.ravel())
            self.img.update()
            self.img.update_tag()
            if self.view3d_area:
                self.view3d_area.tag_redraw()
            self.buffer_dirty = False
            self.last_gpu_sync_time = now

    def paint_stroke_to_buffer(self, context, uv_curr, pressure=1.0):
        s = context.scene.subwater_settings
        color = self.get_active_brush_color(context)
        eff_flow = s.strength * max(pressure, 0.1)
        luma_map = get_cached_luminance_map(s.brush_texture) if s.use_brush_texture else None

        if self.last_uv is not None:
            dist_uv = (uv_curr - self.last_uv).length
            step_uv = max(s.uv_radius / float(self.img_w) * 0.25, 0.002)
            if dist_uv < 0.25 and dist_uv > step_uv:
                steps = min(int(dist_uv / step_uv), 25)
                for i in range(1, steps + 1):
                    t = i / float(steps)
                    uv_interp = self.last_uv.lerp(uv_curr, t)
                    dab_angle = (time.perf_counter() * 137.5 + i * 2.4) % (2.0 * math.pi)
                    apply_subtractive_watercolor_dab(
                        self.pixel_buf, uv_interp.x, uv_interp.y,
                        s.uv_radius, color, eff_flow * 0.65,
                        s, luma_map, dab_angle
                    )
            else:
                dab_angle = (time.perf_counter() * 137.5) % (2.0 * math.pi)
                apply_subtractive_watercolor_dab(
                    self.pixel_buf, uv_curr.x, uv_curr.y,
                    s.uv_radius, color, eff_flow,
                    s, luma_map, dab_angle
                )
        else:
            dab_angle = (time.perf_counter() * 137.5) % (2.0 * math.pi)
            apply_subtractive_watercolor_dab(
                self.pixel_buf, uv_curr.x, uv_curr.y,
                s.uv_radius, color, eff_flow,
                s, luma_map, dab_angle
            )

        self.last_uv = uv_curr
        self.current_uv = uv_curr
        self.buffer_dirty = True
        self.flush_buffer_to_gpu(context, force=False)

    @staticmethod
    def draw_cursor(op, context):
        if op.hit_world is None or op.hit_normal is None:
            return
        s = context.scene.subwater_settings
        radius_3d = s.uv_radius * 0.0035
        n = op.hit_normal
        up = Vector((0, 0, 1)) if abs(n.z) < 0.99 else Vector((1, 0, 0))
        t1 = n.cross(up).normalized()
        t2 = n.cross(t1).normalized()

        verts = [
            op.hit_world + n * 0.003 + (t1 * math.cos(2 * math.pi * i / 40) + t2 * math.sin(2 * math.pi * i / 40)) * radius_3d
            for i in range(41)
        ]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": verts})
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.2)
        c = op.get_active_brush_color(context)
        shader.bind()
        shader.uniform_float("color", (c[0], c[1], c[2], 0.95))
        batch.draw(shader)
        gpu.state.blend_set('NONE')

    def modal(self, context, event):
        if not SubWaterRuntimeState.active or not context.active_object:
            self.finish(context)
            return {'FINISHED'}

        if event.type == 'ESC' and event.value == 'PRESS':
            SubWaterRuntimeState.active = False
            self.finish(context)
            return {'FINISHED'}

        if event.type == 'TIMER':
            if self.is_painting and self.current_uv is not None:
                s = context.scene.subwater_settings
                if s.wet_bleed > 0.05:
                    luma_map = get_cached_luminance_map(s.brush_texture) if s.use_brush_texture else None
                    apply_subtractive_watercolor_dab(
                        self.pixel_buf, self.current_uv.x, self.current_uv.y,
                        s.uv_radius, self.get_active_brush_color(context),
                        s.strength * 0.08, s, luma_map, 0.0
                    )
                    self.buffer_dirty = True
            self.flush_buffer_to_gpu(context, force=False)
            return {'RUNNING_MODAL'}

        # Allow seamless UI interaction without pressing ESC
        if not self.is_mouse_in_viewport_window(context, event):
            self.hit_world = None
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                self.is_painting = False
                self.last_uv = None
                self.current_uv = None
                self.flush_buffer_to_gpu(context, force=True)
            return {'PASS_THROUGH'}

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} or event.alt or event.shift:
            return {'PASS_THROUGH'}

        if event.type == 'LEFT_BRACKET' and event.value == 'PRESS':
            context.scene.subwater_settings.uv_radius = max(2, int(context.scene.subwater_settings.uv_radius * 0.85))
            self.view3d_area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'RIGHT_BRACKET' and event.value == 'PRESS':
            context.scene.subwater_settings.uv_radius = min(300, int(context.scene.subwater_settings.uv_radius * 1.15) + 1)
            self.view3d_area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            uv = self.raycast_uv(context, event)
            if self.is_painting and uv is not None:
                pressure = getattr(event, "pressure", 1.0)
                self.paint_stroke_to_buffer(context, uv, pressure)
            elif self.view3d_area:
                self.view3d_area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                uv = self.raycast_uv(context, event)
                if uv is not None:
                    self.img.pixels.foreach_get(self.pixel_buf.ravel())
                    self.is_painting = True
                    self.last_uv = None
                    pressure = getattr(event, "pressure", 1.0)
                    self.paint_stroke_to_buffer(context, uv, pressure)
                    self.flush_buffer_to_gpu(context, force=True)
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            elif event.value == 'RELEASE':
                self.is_painting = False
                self.last_uv = None
                self.current_uv = None
                self.flush_buffer_to_gpu(context, force=True)
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if SubWaterRuntimeState.active:
            SubWaterRuntimeState.active = False
            return {'FINISHED'}

        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a Mesh object first.")
            return {'CANCELLED'}

        if context.mode != 'PAINT_TEXTURE':
            bpy.ops.object.mode_set(mode='TEXTURE_PAINT')

        self.img = ensure_texture_paint_canvas(context, obj, force_reset=False)
        self.img_w, self.img_h = self.img.size[0], self.img.size[1]
        self.pixel_buf = np.zeros((self.img_h, self.img_w, 4), dtype=np.float32)
        self.img.pixels.foreach_get(self.pixel_buf.ravel())

        self.build_bvh_and_uv_cache(context, obj)

        self.view3d_area = context.area
        self.view3d_space = context.space_data
        self.win_region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)

        SubWaterRuntimeState.active = True
        self.buffer_dirty = False
        self.last_gpu_sync_time = time.perf_counter()

        self._timer = context.window_manager.event_timer_add(0.01, window=context.window)
        self._handle_draw = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_cursor, (self, context), 'WINDOW', 'POST_VIEW'
        )
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "SubWater Live: LMB = Paint | Change Color/Texture in UI Anytime | [ / ] = Size | ESC = Stop"
        )
        return {'RUNNING_MODAL'}

    def finish(self, context):
        SubWaterRuntimeState.active = False
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._handle_draw:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_draw, 'WINDOW')
            self._handle_draw = None
        context.workspace.status_text_set(None)
        if self.view3d_area:
            self.view3d_area.tag_redraw()


class IMAGE_OT_subwater_clear_paper(bpy.types.Operator):
    """Clears the active texture canvas back to warm watercolor paper"""
    bl_idname = "image.subwater_clear_paper"
    bl_label = "Clear to Watercolor Paper"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}
        ensure_texture_paint_canvas(context, obj, force_reset=True)
        return {'FINISHED'}


class PAINT_OT_subwater_preset_color(bpy.types.Operator):
    """Quick pigment swatch selector"""
    bl_idname = "paint.subwater_preset_color"
    bl_label = "Set Pigment Swatch"
    color: bpy.props.FloatVectorProperty(size=3)

    def execute(self, context):
        ts = context.tool_settings
        if ts.image_paint and ts.image_paint.brush:
            ts.image_paint.brush.color = self.color
        if ts.unified_paint_settings:
            ts.unified_paint_settings.color = self.color
        return {'FINISHED'}


# --- N-PANEL UI ---

class VIEW3D_PT_subwater_texpaint_panel(bpy.types.Panel):
    bl_label = "SubWater Texture Paint"
    bl_idname = "VIEW3D_PT_subwater_texpaint_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SubWater"

    def draw(self, context):
        layout = self.layout
        s = context.scene.subwater_settings
        ts = context.tool_settings

        col = layout.column(align=True)
        col.scale_y = 1.5
        if SubWaterRuntimeState.active:
            col.operator(
                "paint.subwater_texpaint_modal",
                text="SubWater Brush: ACTIVE (Click to Stop)",
                icon='REC', depress=True
            )
        else:
            col.operator(
                "paint.subwater_texpaint_modal",
                text="Start SubWater Brush (Texture Paint)",
                icon='BRUSH_DATA'
            )

        # Real-time Sync Interval Control (10-100ms)
        box_sync = layout.box()
        box_sync.label(text="Real-Time Viewport Sync", icon='TIME')
        box_sync.prop(s, "update_interval_ms", slider=True)

        # Live Native Color Picker
        box = layout.box()
        box.label(text="Pigment Color (Live Switch)", icon='COLOR')
        if ts.image_paint and ts.image_paint.brush:
            box.template_color_picker(ts.image_paint.brush, "color", value_slider=True)
            box.prop(ts.image_paint.brush, "color", text="")

        row = box.row(align=True)
        p1 = row.operator("paint.subwater_preset_color", text="Cyan")
        p1.color = (0.02, 0.55, 0.92)
        p2 = row.operator("paint.subwater_preset_color", text="Yellow")
        p2.color = (0.98, 0.82, 0.04)
        p3 = row.operator("paint.subwater_preset_color", text="Crimson")
        p3.color = (0.85, 0.05, 0.18)
        p4 = row.operator("paint.subwater_preset_color", text="Ultramarine")
        p4.color = (0.06, 0.14, 0.82)

        # NEW: Brush Luminance Alpha Texture Channel
        box_tex = layout.box()
        row_hdr = box_tex.row()
        row_hdr.prop(s, "use_brush_texture", text="Luminance Alpha Texture", icon='TEXTURE')
        if s.use_brush_texture:
            box_tex.template_ID(s, "brush_texture", open="image.open")
            box_tex.operator("image.subwater_generate_grain_tex", icon='SPARKLES')
            box_tex.prop(s, "tex_mapping")
            box_tex.prop(s, "tex_strength", slider=True)
            box_tex.prop(s, "tex_contrast", slider=True)
            if s.tex_mapping == 'STAMP':
                box_tex.prop(s, "tex_jitter_angle")
            else:
                box_tex.prop(s, "tex_scale", slider=True)
            box_tex.prop(s, "tex_invert")

        # Brush Dynamics
        box2 = layout.box()
        box2.label(text="Subtractive Watercolor Physics", icon='MOD_FLUIDSIM')
        box2.prop(s, "uv_radius", slider=True)
        box2.prop(s, "strength", slider=True)
        box2.prop(s, "water_dilution", slider=True)
        box2.prop(s, "edge_darkening", slider=True)
        box2.prop(s, "wet_bleed", slider=True)

        # Canvas Management
        box3 = layout.box()
        box3.label(text="UV Texture Canvas", icon='IMAGE_DATA')
        box3.prop(s, "canvas_size", text="Res")
        box3.operator("image.subwater_clear_paper", icon='FILE_REFRESH')


classes = (
    SubWaterSettings,
    IMAGE_OT_subwater_generate_grain_tex,
    PAINT_OT_subwater_texpaint_modal,
    IMAGE_OT_subwater_clear_paper,
    PAINT_OT_subwater_preset_color,
    VIEW3D_PT_subwater_texpaint_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.subwater_settings = bpy.props.PointerProperty(type=SubWaterSettings)


def unregister():
    SubWaterRuntimeState.active = False
    del bpy.types.Scene.subwater_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()