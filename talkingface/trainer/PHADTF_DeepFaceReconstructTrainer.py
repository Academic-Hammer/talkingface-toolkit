#==============================================================================================================================================================
#mesh_renderer_test.py

import math
import os

import numpy as np
import tensorflow as tf



class RenderTest(tf.test.TestCase):

  def setUp(self):
    self.test_data_directory = (
        'path/to/test_data/')

    tf.reset_default_graph()
    # Set up a basic cube centered at the origin, with vertex normals pointing
    # outwards along the line from the origin to the cube vertices:
    self.cube_vertices = tf.constant(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=tf.float32)
    self.cube_normals = tf.nn.l2_normalize(self.cube_vertices, dim=1)
    self.cube_triangles = tf.constant(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=tf.int32)

  def testRendersSimpleCube(self):
    """Renders a simple cube to test the full forward pass.

    Verifies the functionality of both the custom kernel and the python wrapper.
    """

    model_transforms = euler_matrices(
        [[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]])[:, :3, :3]

    vertices_world_space = tf.matmul(
        tf.stack([self.cube_vertices, self.cube_vertices]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([self.cube_normals, self.cube_normals]),
        model_transforms,
        transpose_b=True)

    # camera position:
    eye = tf.constant(2 * [[0.0, 0.0, 6.0]], dtype=tf.float32)
    center = tf.constant(2 * [[0.0, 0.0, 0.0]], dtype=tf.float32)
    world_up = tf.constant(2 * [[0.0, 1.0, 0.0]], dtype=tf.float32)
    image_width = 640
    image_height = 480
    light_positions = tf.constant([[[0.0, 0.0, 6.0]], [[0.0, 0.0, 6.0]]])
    light_intensities = tf.ones([2, 1, 3], dtype=tf.float32)
    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)

    rendered = mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)

    with self.test_session() as sess:
      images = sess.run(rendered, feed_dict={})
      for image_id in range(images.shape[0]):
        target_image_name = 'Gray_Cube_%i.png' % image_id
        baseline_image_path = os.path.join(self.test_data_directory,
                                           target_image_name)
        expect_image_file_and_render_are_near(
            self, sess, baseline_image_path, images[image_id, :, :, :])

  def testComplexShading(self):
    """Tests specular highlights, colors, and multiple lights per image."""
    # rotate the cube for the test:
    model_transforms = euler_matrices(
        [[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]])[:, :3, :3]

    vertices_world_space = tf.matmul(
        tf.stack([self.cube_vertices, self.cube_vertices]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([self.cube_normals, self.cube_normals]),
        model_transforms,
        transpose_b=True)

    # camera position:
    eye = tf.constant([[0.0, 0.0, 6.0], [0., 0.2, 18.0]], dtype=tf.float32)
    center = tf.constant([[0.0, 0.0, 0.0], [0.1, -0.1, 0.1]], dtype=tf.float32)
    world_up = tf.constant(
        [[0.0, 1.0, 0.0], [0.1, 1.0, 0.15]], dtype=tf.float32)
    fov_y = tf.constant([40., 13.3], dtype=tf.float32)
    near_clip = tf.constant(0.1, dtype=tf.float32)
    far_clip = tf.constant(25.0, dtype=tf.float32)
    image_width = 640
    image_height = 480
    light_positions = tf.constant([[[0.0, 0.0, 6.0], [1.0, 2.0, 6.0]],
                                   [[0.0, -2.0, 4.0], [1.0, 3.0, 4.0]]])
    light_intensities = tf.constant(
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[2.0, 0.0, 1.0], [0.0, 2.0,
                                                                1.0]]],
        dtype=tf.float32)
    # pyformat: disable
    vertex_diffuse_colors = tf.constant(2*[[[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [1.0, 1.0, 1.0],
                                            [1.0, 1.0, 0.0],
                                            [1.0, 0.0, 1.0],
                                            [0.0, 1.0, 1.0],
                                            [0.5, 0.5, 0.5]]],
                                        dtype=tf.float32)
    vertex_specular_colors = tf.constant(2*[[[0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 1.0, 1.0],
                                             [1.0, 1.0, 0.0],
                                             [1.0, 0.0, 1.0],
                                             [0.0, 1.0, 1.0],
                                             [0.5, 0.5, 0.5],
                                             [1.0, 0.0, 0.0]]],
                                         dtype=tf.float32)
    # pyformat: enable
    shininess_coefficients = 6.0 * tf.ones([2, 8], dtype=tf.float32)
    ambient_color = tf.constant(
        [[0., 0., 0.], [0.1, 0.1, 0.2]], dtype=tf.float32)
    renders = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height, vertex_specular_colors,
        shininess_coefficients, ambient_color, fov_y, near_clip, far_clip)
    tonemapped_renders = tf.concat(
        [
            mesh_renderer.tone_mapper(renders[:, :, :, 0:3], 0.7),
            renders[:, :, :, 3:4]
        ],
        axis=3)

    # Check that shininess coefficient broadcasting works by also rendering
    # with a scalar shininess coefficient, and ensuring the result is identical:
    broadcasted_renders = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height, vertex_specular_colors,
        6.0, ambient_color, fov_y, near_clip, far_clip)
    tonemapped_broadcasted_renders = tf.concat(
        [
            mesh_renderer.tone_mapper(broadcasted_renders[:, :, :, 0:3], 0.7),
            broadcasted_renders[:, :, :, 3:4]
        ],
        axis=3)

    with self.test_session() as sess:
      images, broadcasted_images = sess.run(
          [tonemapped_renders, tonemapped_broadcasted_renders], feed_dict={})

      for image_id in range(images.shape[0]):
        target_image_name = 'Colored_Cube_%i.png' % image_id
        baseline_image_path = os.path.join(self.test_data_directory,
                                           target_image_name)
        expect_image_file_and_render_are_near(
            self, sess, baseline_image_path, images[image_id, :, :, :])
        expect_image_file_and_render_are_near(
            self, sess, baseline_image_path,
            broadcasted_images[image_id, :, :, :])

  def testFullRenderGradientComputation(self):
    """Verifies the Jacobian matrix for the entire renderer.

    This ensures correct gradients are propagated backwards through the entire
    process, not just through the rasterization kernel. Uses the simple cube
    forward pass.
    """
    image_height = 21
    image_width = 28

    # rotate the cube for the test:
    model_transforms = euler_matrices(
        [[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]])[:, :3, :3]

    vertices_world_space = tf.matmul(
        tf.stack([self.cube_vertices, self.cube_vertices]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([self.cube_normals, self.cube_normals]),
        model_transforms,
        transpose_b=True)

    # camera position:
    eye = tf.constant([0.0, 0.0, 6.0], dtype=tf.float32)
    center = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    world_up = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)

    # Scene has a single light from the viewer's eye.
    light_positions = tf.expand_dims(tf.stack([eye, eye], axis=0), axis=1)
    light_intensities = tf.ones([2, 1, 3], dtype=tf.float32)

    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)

    rendered = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)

    with self.test_session():
      theoretical, numerical = tf.test.compute_gradient(
          self.cube_vertices, (8, 3),
          rendered, (2, image_height, image_width, 4),
          x_init_value=self.cube_vertices.eval(),
          delta=1e-3)
      jacobians_match, message = (
          check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.01))
      self.assertTrue(jacobians_match, message)

  def testThatCubeRotates(self):
    """Optimize a simple cube's rotation using pixel loss.

    The rotation is represented as static-basis euler angles. This test checks
    that the computed gradients are useful.
    """
    image_height = 480
    image_width = 640
    initial_euler_angles = [[0.0, 0.0, 0.0]]

    euler_angles = tf.Variable(initial_euler_angles)
    model_rotation = euler_matrices(euler_angles)[0, :3, :3]

    vertices_world_space = tf.reshape(
        tf.matmul(self.cube_vertices, model_rotation, transpose_b=True),
        [1, 8, 3])

    normals_world_space = tf.reshape(
        tf.matmul(self.cube_normals, model_rotation, transpose_b=True),
        [1, 8, 3])

    # camera position:
    eye = tf.constant([[0.0, 0.0, 6.0]], dtype=tf.float32)
    center = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    world_up = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)

    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)
    light_positions = tf.reshape(eye, [1, 1, 3])
    light_intensities = tf.ones([1, 1, 3], dtype=tf.float32)

    render = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)
    render = tf.reshape(render, [image_height, image_width, 4])

    # Pick the desired cube rotation for the test:
    test_model_rotation = euler_matrices([[-20.0, 0.0,
                                                        60.0]])[0, :3, :3]

    desired_vertex_positions = tf.reshape(
        tf.matmul(self.cube_vertices, test_model_rotation, transpose_b=True),
        [1, 8, 3])
    desired_normals = tf.reshape(
        tf.matmul(self.cube_normals, test_model_rotation, transpose_b=True),
        [1, 8, 3])
    desired_render = mesh_renderer.mesh_renderer(
        desired_vertex_positions, self.cube_triangles, desired_normals,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)
    desired_render = tf.reshape(desired_render, [image_height, image_width, 4])

    loss = tf.reduce_mean(tf.abs(render - desired_render))
    optimizer = tf.train.MomentumOptimizer(0.7, 0.1)
    grad = tf.gradients(loss, [euler_angles])
    grad, _ = tf.clip_by_global_norm(grad, 1.0)
    opt_func = optimizer.apply_gradients([(grad[0], euler_angles)])

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(35):
        sess.run([loss, opt_func])
      final_image, desired_image = sess.run([render, desired_render])

      target_image_name = 'Gray_Cube_0.png'
      baseline_image_path = os.path.join(self.test_data_directory,
                                         target_image_name)
      expect_image_file_and_render_are_near(
          self, sess, baseline_image_path, desired_image)
      expect_image_file_and_render_are_near(
          self,
          sess,
          baseline_image_path,
          final_image,
          max_outlier_fraction=0.01,
          pixel_error_threshold=0.04)


if __name__ == '__main__':
  tf.test.main()



#==============================================================================================================================================================
#camera_utils.py
def euler_matrices(angles):
  s = tf.sin(angles)
  c = tf.cos(angles)
  # Rename variables for readability in the matrix definition below.
  c0, c1, c2 = (c[:, 0], c[:, 1], c[:, 2])
  s0, s1, s2 = (s[:, 0], s[:, 1], s[:, 2])

  zeros = tf.zeros_like(s[:, 0])
  ones = tf.ones_like(s[:, 0])

  # pyformat: disable
  flattened = tf.concat(
      [
          c2 * c1, c2 * s1 * s0 - c0 * s2, s2 * s0 + c2 * c0 * s1, zeros,
          c1 * s2, c2 * c0 + s2 * s1 * s0, c0 * s2 * s1 - c2 * s0, zeros,
          -s1, c1 * s0, c1 * c0, zeros,
          zeros, zeros, zeros, ones
      ],
      axis=0)
  # pyformat: enable
  reshaped = tf.reshape(flattened, [4, 4, -1])
  return tf.transpose(reshaped, [2, 0, 1])

def perspective(aspect_ratio, fov_y, near_clip, far_clip):
  # The multiplication of fov_y by pi/360.0 simultaneously converts to radians
  # and adds the half-angle factor of .5.
  focal_lengths_y = 1.0 / tf.tan(fov_y * (math.pi / 360.0))
  depth_range = far_clip - near_clip
  p_22 = -(far_clip + near_clip) / depth_range
  p_23 = -2.0 * (far_clip * near_clip / depth_range)

  zeros = tf.zeros_like(p_23, dtype=tf.float32)
  # pyformat: disable
  perspective_transform = tf.concat(
      [
          focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
          zeros, focal_lengths_y, zeros, zeros,
          zeros, zeros, p_22, p_23,
          zeros, zeros, -tf.ones_like(p_23, dtype=tf.float32), zeros
      ], axis=0)
  # pyformat: enable
  perspective_transform = tf.reshape(perspective_transform, [4, 4, -1])
  return tf.transpose(perspective_transform, [2, 0, 1])


def look_at(eye, center, world_up):
  batch_size = center.shape[0].value
  vector_degeneracy_cutoff = 1e-6
  forward = center - eye
  forward_norm = tf.norm(forward, ord='euclidean', axis=1, keepdims=True)
  #tf.assert_greater(
  #    forward_norm,
  #    vector_degeneracy_cutoff,
  #    message='Camera matrix is degenerate because eye and center are close.')
  forward = tf.divide(forward, forward_norm)

  to_side = tf.linalg.cross(forward, world_up)
  to_side_norm = tf.norm(to_side, ord='euclidean', axis=1, keepdims=True)
  #tf.assert_greater(
  #    to_side_norm,
  #    vector_degeneracy_cutoff,
  #    message='Camera matrix is degenerate because up and gaze are close or'
  #    'because up is degenerate.')
  to_side = tf.divide(to_side, to_side_norm)
  cam_up = tf.linalg.cross(to_side, forward)

  w_column = tf.constant(
      batch_size * [[0., 0., 0., 1.]], dtype=tf.float32)  # [batch_size, 4]
  w_column = tf.reshape(w_column, [batch_size, 4, 1])
  view_rotation = tf.stack(
      [to_side, cam_up, -forward,
       tf.zeros_like(to_side, dtype=tf.float32)],
      axis=1)  # [batch_size, 4, 3] matrix
  view_rotation = tf.concat(
      [view_rotation, w_column], axis=2)  # [batch_size, 4, 4]

  identity_batch = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
  view_translation = tf.concat([identity_batch, tf.expand_dims(-eye, 2)], 2)
  view_translation = tf.concat(
      [view_translation,
       tf.reshape(w_column, [batch_size, 1, 4])], 1)
  camera_matrices = tf.matmul(view_rotation, view_translation)
  return camera_matrices

def transform_homogeneous(matrices, vertices):
  if len(matrices.shape) != 3:
    raise ValueError(
        'matrices must have 3 dimensions (missing batch dimension?)')
  if len(vertices.shape) != 3:
    raise ValueError(
        'vertices must have 3 dimensions (missing batch dimension?)')
  homogeneous_coord = tf.ones(
      [tf.shape(vertices)[0], tf.shape(vertices)[1], 1], dtype=tf.float32)
  vertices_homogeneous = tf.concat([vertices, homogeneous_coord], 2)

  return tf.matmul(vertices_homogeneous, matrices, transpose_b=True)

#==============================================================================================================================================================
#mesh_renderer.py
def phong_shader(normals,
                 alphas,
                 pixel_positions,
                 light_positions,
                 light_intensities,
                 diffuse_colors=None,
                 camera_position=None,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None):
  batch_size, image_height, image_width = [s.value for s in normals.shape[:-1]]
  light_count = light_positions.shape[1].value
  pixel_count = image_height * image_width
  # Reshape all values to easily do pixelwise computations:
  normals = tf.reshape(normals, [batch_size, -1, 3])
  alphas = tf.reshape(alphas, [batch_size, -1, 1])
  diffuse_colors = tf.reshape(diffuse_colors, [batch_size, -1, 3])
  if camera_position is not None:
    specular_colors = tf.reshape(specular_colors, [batch_size, -1, 3])

  # Ambient component
  output_colors = tf.zeros([batch_size, image_height * image_width, 3])
  if ambient_color is not None:
    ambient_reshaped = tf.expand_dims(ambient_color, axis=1)
    output_colors = tf.add(output_colors, ambient_reshaped * diffuse_colors)

  # Diffuse component
  pixel_positions = tf.reshape(pixel_positions, [batch_size, -1, 3])
  per_light_pixel_positions = tf.stack(
      [pixel_positions] * light_count,
      axis=1)  # [batch_size, light_count, pixel_count, 3]
  directions_to_lights = tf.nn.l2_normalize(
      tf.expand_dims(light_positions, axis=2) - per_light_pixel_positions,
      axis=3)  # [batch_size, light_count, pixel_count, 3]
  # The specular component should only contribute when the light and normal
  # face one another (i.e. the dot product is nonnegative):
  normals_dot_lights = tf.clip_by_value(
      tf.reduce_sum(
          tf.expand_dims(normals, axis=1) * directions_to_lights, axis=3), 0.0,
      1.0)  # [batch_size, light_count, pixel_count]
  diffuse_output = tf.expand_dims(
      diffuse_colors, axis=1) * tf.expand_dims(
          normals_dot_lights, axis=3) * tf.expand_dims(
              light_intensities, axis=2)
  diffuse_output = tf.reduce_sum(
      diffuse_output, axis=1)  # [batch_size, pixel_count, 3]
  output_colors = tf.add(output_colors, diffuse_output)

  # Specular component
  if camera_position is not None:
    camera_position = tf.reshape(camera_position, [batch_size, 1, 3])
    mirror_reflection_direction = tf.nn.l2_normalize(
        2.0 * tf.expand_dims(normals_dot_lights, axis=3) * tf.expand_dims(
            normals, axis=1) - directions_to_lights,
        dim=3)
    direction_to_camera = tf.nn.l2_normalize(
        camera_position - pixel_positions, dim=2)
    reflection_direction_dot_camera_direction = tf.reduce_sum(
        tf.expand_dims(direction_to_camera, axis=1) *
        mirror_reflection_direction,
        axis=3)
    # The specular component should only contribute when the reflection is
    # external:
    reflection_direction_dot_camera_direction = tf.clip_by_value(
        tf.nn.l2_normalize(reflection_direction_dot_camera_direction, dim=2),
        0.0, 1.0)
    # The specular component should also only contribute when the diffuse
    # component contributes:
    reflection_direction_dot_camera_direction = tf.where(
        normals_dot_lights != 0.0, reflection_direction_dot_camera_direction,
        tf.zeros_like(
            reflection_direction_dot_camera_direction, dtype=tf.float32))
    # Reshape to support broadcasting the shininess coefficient, which rarely
    # varies per-vertex:
    reflection_direction_dot_camera_direction = tf.reshape(
        reflection_direction_dot_camera_direction,
        [batch_size, light_count, image_height, image_width])
    shininess_coefficients = tf.expand_dims(shininess_coefficients, axis=1)
    specularity = tf.reshape(
        tf.pow(reflection_direction_dot_camera_direction,
               shininess_coefficients),
        [batch_size, light_count, pixel_count, 1])
    specular_output = tf.expand_dims(
        specular_colors, axis=1) * specularity * tf.expand_dims(
            light_intensities, axis=2)
    specular_output = tf.reduce_sum(specular_output, axis=1)
    output_colors = tf.add(output_colors, specular_output)
  rgb_images = tf.reshape(output_colors,
                          [batch_size, image_height, image_width, 3])
  alpha_images = tf.reshape(alphas, [batch_size, image_height, image_width, 1])
  valid_rgb_values = tf.concat(3 * [alpha_images > 0.5], axis=3)
  rgb_images = tf.where(valid_rgb_values, rgb_images,
                        tf.zeros_like(rgb_images, dtype=tf.float32))
  return tf.reverse(tf.concat([rgb_images, alpha_images], axis=3), axis=[1])


def mesh_renderer(vertices,
                  triangles,
                  normals,
                  diffuse_colors,
                  camera_position,
                  camera_lookat,
                  camera_up,
                  light_positions,
                  light_intensities,
                  image_width,
                  image_height,
                  specular_colors=None,
                  shininess_coefficients=None,
                  ambient_color=None,
                  fov_y=40.0,
                  near_clip=0.01,
                  far_clip=10.0):
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if len(light_positions.shape) != 3:
    raise ValueError(
        'Light_positions must have shape [batch_size, light_count, 3].')
  if len(light_intensities.shape) != 3:
    raise ValueError(
        'Light_intensities must have shape [batch_size, light_count, 3].')
  if len(diffuse_colors.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
  if (ambient_color is not None and
      ambient_color.get_shape().as_list() != [batch_size, 3]):
    raise ValueError('Ambient_color must have shape [batch_size, 3].')
  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')
  if camera_lookat.get_shape().as_list() == [3]:
    camera_lookat = tf.tile(
        tf.expand_dims(camera_lookat, axis=0), [batch_size, 1])
  elif camera_lookat.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_lookat must have shape [batch_size, 3]')
  if camera_up.get_shape().as_list() == [3]:
    camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [batch_size, 1])
  elif camera_up.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_up must have shape [batch_size, 3]')
  if isinstance(fov_y, float):
    fov_y = tf.constant(batch_size * [fov_y], dtype=tf.float32)
  elif not fov_y.get_shape().as_list():
    fov_y = tf.tile(tf.expand_dims(fov_y, 0), [batch_size])
  elif fov_y.get_shape().as_list() != [batch_size]:
    raise ValueError('Fov_y must be a float, a 0D tensor, or a 1D tensor with'
                     'shape [batch_size]')
  if isinstance(near_clip, float):
    near_clip = tf.constant(batch_size * [near_clip], dtype=tf.float32)
  elif not near_clip.get_shape().as_list():
    near_clip = tf.tile(tf.expand_dims(near_clip, 0), [batch_size])
  elif near_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Near_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if isinstance(far_clip, float):
    far_clip = tf.constant(batch_size * [far_clip], dtype=tf.float32)
  elif not far_clip.get_shape().as_list():
    far_clip = tf.tile(tf.expand_dims(far_clip, 0), [batch_size])
  elif far_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Far_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if specular_colors is not None and shininess_coefficients is None:
    raise ValueError(
        'Specular colors were supplied without shininess coefficients.')
  if shininess_coefficients is not None and specular_colors is None:
    raise ValueError(
        'Shininess coefficients were supplied without specular colors.')
  if specular_colors is not None:
    # Since a 0-D float32 tensor is accepted, also accept a float.
    if isinstance(shininess_coefficients, float):
      shininess_coefficients = tf.constant(
          shininess_coefficients, dtype=tf.float32)
    if len(specular_colors.shape) != 3:
      raise ValueError('The specular colors must have shape [batch_size, '
                       'vertex_count, 3].')
    if len(shininess_coefficients.shape) > 2:
      raise ValueError('The shininess coefficients must have shape at most'
                       '[batch_size, vertex_count].')
    # If we don't have per-vertex coefficients, we can just reshape the
    # input shininess to broadcast later, rather than interpolating an
    # additional vertex attribute:
    if len(shininess_coefficients.shape) < 2:
      vertex_attributes = tf.concat(
          [normals, vertices, diffuse_colors, specular_colors], axis=2)
    else:
      vertex_attributes = tf.concat(
          [
              normals, vertices, diffuse_colors, specular_colors,
              tf.expand_dims(shininess_coefficients, axis=2)
          ],
          axis=2)
  else:
    vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

  camera_matrices = look_at(camera_position, camera_lookat,
                                         camera_up)

  perspective_transforms = perspective(image_width / image_height,
                                                    fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(perspective_transforms, camera_matrices)

  pixel_attributes = rasterize(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [-1] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], axis=3)
  pixel_positions = pixel_attributes[:, :, :, 3:6]
  diffuse_colors = pixel_attributes[:, :, :, 6:9]
  if specular_colors is not None:
    specular_colors = pixel_attributes[:, :, :, 9:12]
    # Retrieve the interpolated shininess coefficients if necessary, or just
    # reshape our input for broadcasting:
    if len(shininess_coefficients.shape) == 2:
      shininess_coefficients = pixel_attributes[:, :, :, 12]
    else:
      shininess_coefficients = tf.reshape(shininess_coefficients, [-1, 1, 1])

  pixel_mask = tf.cast(tf.reduce_any(diffuse_colors >= 0, axis=3), tf.float32)

  renders = phong_shader(
      normals=pixel_normals,
      alphas=pixel_mask,
      pixel_positions=pixel_positions,
      light_positions=light_positions,
      light_intensities=light_intensities,
      diffuse_colors=diffuse_colors,
      camera_position=camera_position if specular_colors is not None else None,
      specular_colors=specular_colors,
      shininess_coefficients=shininess_coefficients,
      ambient_color=ambient_color)
  return renders

#==============================================================================================================================================================
#rasterize_triangles.py
rasterize_triangles_module = tf.load_op_library(
    #os.path.join(os.environ['TEST_SRCDIR'],
    os.path.join('/home4/yiran/TalkingFace/Pipeline/Deep3DFaceReconstruction',
    'tf_mesh_renderer/mesh_renderer/kernels/rasterize_triangles_kernel.so'))

def rasterize(world_space_vertices, attributes, triangles, camera_matrices,
              image_width, image_height, background_value):
  clip_space_vertices = transform_homogeneous(
      camera_matrices, world_space_vertices)
  return rasterize_clip_space(clip_space_vertices, attributes, triangles,
                              image_width, image_height, background_value)


def rasterize_clip_space(clip_space_vertices, attributes, triangles,
                         image_width, image_height, background_value):
  if not image_width > 0:
    raise ValueError('Image width must be > 0.')
  if not image_height > 0:
    raise ValueError('Image height must be > 0.')
  if len(clip_space_vertices.shape) != 3:
    raise ValueError('The vertex buffer must be 3D.')

  vertex_count = clip_space_vertices.shape[1].value

  batch_size = tf.shape(clip_space_vertices)[0]
  
  per_image_barycentric_coordinates = tf.TensorArray(dtype=tf.float32,
    size=batch_size)
  per_image_vertex_ids = tf.TensorArray(dtype=tf.int32, size=batch_size)

  def batch_loop_condition(b, *args):
    return b < batch_size

  def batch_loop_iteration(b, per_image_barycentric_coordinates,
    per_image_vertex_ids):
    barycentric_coords, triangle_ids, _ = (
        rasterize_triangles_module.rasterize_triangles(
            clip_space_vertices[b, :, :], triangles, image_width,
            image_height))
    per_image_barycentric_coordinates = \
      per_image_barycentric_coordinates.write(
        b, tf.reshape(barycentric_coords, [-1, 3]))

    vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
    reindexed_ids = tf.add(vertex_ids, b * clip_space_vertices.shape[1].value)
    per_image_vertex_ids = per_image_vertex_ids.write(b, reindexed_ids)

    return b+1, per_image_barycentric_coordinates, per_image_vertex_ids

  b = tf.constant(0)
  _, per_image_barycentric_coordinates, per_image_vertex_ids = tf.while_loop(
    batch_loop_condition, batch_loop_iteration,
    [b, per_image_barycentric_coordinates, per_image_vertex_ids])

  barycentric_coordinates = tf.reshape(
    per_image_barycentric_coordinates.stack(), [-1, 3])
  vertex_ids = tf.reshape(per_image_vertex_ids.stack(), [-1, 3])

  # Indexes with each pixel's clip-space triangle's extrema (the pixel's
  # 'corner points') ids to get the relevant properties for deferred shading.
  flattened_vertex_attributes = tf.reshape(attributes,
                                           [batch_size * vertex_count, -1])
  corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

  # Computes the pixel attributes by interpolating the known attributes at the
  # corner points of the triangle interpolated with the barycentric coordinates.
  weighted_vertex_attributes = tf.multiply(
      corner_attributes, tf.expand_dims(barycentric_coordinates, axis=2))
  summed_attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
  attribute_images = tf.reshape(summed_attributes,
                                [batch_size, image_height, image_width, -1])

  # Barycentric coordinates should approximately sum to one where there is
  # rendered geometry, but be exactly zero where there is not.
  alphas = tf.clip_by_value(
      tf.reduce_sum(2.0 * barycentric_coordinates, axis=1), 0.0, 1.0)
  alphas = tf.reshape(alphas, [batch_size, image_height, image_width, 1])

  attributes_with_background = (
      alphas * attribute_images + (1.0 - alphas) * background_value)

  return attributes_with_background

@tf.RegisterGradient('RasterizeTriangles')
def _rasterize_triangles_grad(op, df_dbarys, df_dids, df_dz):
  # Gradients are only supported for barycentric coordinates. Gradients for the
  # z-buffer are not currently implemented. If you need gradients w.r.t. z,
  # include z as a vertex attribute when calling rasterize_triangles.
  del df_dids, df_dz
  return rasterize_triangles_module.rasterize_triangles_grad(
      op.inputs[0], op.inputs[1], op.outputs[0], op.outputs[1], df_dbarys,
      op.get_attr('image_width'), op.get_attr('image_height')), None

#==============================================================================================================================================================
#test_utils.py

import os
import numpy as np
import tensorflow as tf


def check_jacobians_are_nearly_equal(theoretical,
                                     numerical,
                                     outlier_relative_error_threshold,
                                     max_outlier_fraction,
                                     include_jacobians_in_error_message=False):
  outlier_gradients = np.abs(
      numerical - theoretical) / numerical > outlier_relative_error_threshold
  outlier_fraction = np.count_nonzero(outlier_gradients) / np.prod(
      numerical.shape[:2])
  jacobians_match = outlier_fraction <= max_outlier_fraction

  message = (
      ' %f of theoretical gradients are relative outliers, but the maximum'
      ' allowable fraction is %f ' % (outlier_fraction, max_outlier_fraction))
  if include_jacobians_in_error_message:
    # the gradient_checker convention is the typical Jacobian transposed:
    message += ('\nNumerical Jacobian:\n%s\nTheoretical Jacobian:\n%s' %
                (repr(numerical.T), repr(theoretical.T)))
  return jacobians_match, message


def expect_image_file_and_render_are_near(test_instance,
                                          sess,
                                          baseline_path,
                                          result_image,
                                          max_outlier_fraction=0.001,
                                          pixel_error_threshold=0.01):
  baseline_bytes = open(baseline_path, 'rb').read()
  baseline_image = sess.run(tf.image.decode_png(baseline_bytes))

  test_instance.assertEqual(baseline_image.shape, result_image.shape,
                            'Image shapes %s and %s do not match.' %
                            (baseline_image.shape, result_image.shape))

  result_image = np.clip(result_image, 0., 1.).copy(order='C')
  baseline_image = baseline_image.astype(float) / 255.0

  outlier_channels = (np.abs(baseline_image - result_image) >
                      pixel_error_threshold)
  outlier_pixels = np.any(outlier_channels, axis=2)
  outlier_count = np.count_nonzero(outlier_pixels)
  outlier_fraction = outlier_count / np.prod(baseline_image.shape[:2])
  images_match = outlier_fraction <= max_outlier_fraction

  outputs_dir = "/tmp" #os.environ["TEST_TMPDIR"]
  base_prefix = os.path.splitext(os.path.basename(baseline_path))[0]
  result_output_path = os.path.join(outputs_dir, base_prefix + "_result.png")

  message = ('%s does not match. (%f of pixels are outliers, %f is allowed.). '
             'Result image written to %s' %
             (baseline_path, outlier_fraction, max_outlier_fraction, result_output_path))

  if not images_match:
    result_bytes = sess.run(tf.image.encode_png(result_image*255.0))
    with open(result_output_path, 'wb') as output_file:
      output_file.write(result_bytes)

  test_instance.assertTrue(images_match, msg=message)
