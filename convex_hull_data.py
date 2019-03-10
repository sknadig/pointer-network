from scipy.spatial import ConvexHull

def convex_hull():
        for i in range(1000):
            points = np.random.rand(10, 2)
            hull_indices = ConvexHull(points=points)
            target_indices = hull_indices.vertices
            targets = points[target_indices]

            points = tf.convert_to_tensor([points])
            points = tf.squeeze(points, axis=0)

            targets = tf.convert_to_tensor(targets)
            target_indices = tf.convert_to_tensor(target_indices)

            targets = tf.pad(targets, tf.constant([[0,max_seq_len-targets.shape[0]], [0,0]]), constant_values=-1)
            target_indices = tf.pad(target_indices, tf.constant([[0,max_seq_len-target_indices.shape[0]]]), constant_values=-1)


            targets = tf.convert_to_tensor(targets)
            target_indices = tf.convert_to_tensor(target_indices)

            yield points, targets, target_indices