import pyglet
import trimesh
import numpy as np
from PIL import Image

import proxies.errors as errors


def trimesh_render(mesh, resolution=(1280, 720), mode="L", angle_y=0, angle_x=-35.0):
    try:
        try:
            scene = mesh.scene()
        except AttributeError:
            scene = mesh

        # Get the initial camera transform (identity)
        camera_old = np.eye(4)

        # Move the camera back a little
        mat = trimesh.transformations.translation_matrix([0, 0, 1.5])
        camera_old = np.dot(mat, camera_old)

        # Orient the camera so its facing slightly down
        mat = trimesh.transformations.rotation_matrix(
            angle=np.radians(angle_x), direction=[1, 0, 0], point=scene.centroid
        )
        camera_old = np.dot(mat, camera_old)

        mat = trimesh.transformations.rotation_matrix(
            angle=np.radians(angle_y), direction=[0, 1, 0], point=scene.centroid
        )
        camera_old = np.dot(mat, camera_old)

        # Apply the transform
        scene.graph[scene.camera.name] = camera_old

        data = Image.open(
            trimesh.util.wrap_as_stream(scene.save_image(resolution=resolution))
        )
        return np.array(data.convert(mode))
    except pyglet.canvas.xlib.NoSuchDisplayException:
        raise errors.NoDisplayError


def process(f_in, f_out, angle=0, resolution=(640, 640)):
    mesh = trimesh.load(f_in)
    img = trimesh_render(mesh, resolution=resolution, mode="RGB", angle_y=angle)
    Image.fromarray(img).save(f_out)


if __name__ == "__main__":
    process("test/model_normalized.obj", "test.png", angle=0)
