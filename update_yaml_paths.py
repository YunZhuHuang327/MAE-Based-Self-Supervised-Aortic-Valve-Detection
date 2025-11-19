"""
Update paths in args.yaml to match current environment.

Only updates paths, preserves all training parameters.
"""

import yaml
from pathlib import Path


def update_yaml_paths(
    input_yaml='/DATA1/yunzhu/SSL/args.yaml',
    output_yaml='/DATA1/yunzhu/SSL/args_updated.yaml',
    data_yaml='/DATA1/yunzhu/SSL/valve_detection.yaml',
    project_dir='/DATA1/yunzhu/SSL/valve_training_results'
):
    """
    Update only the path-related fields in YAML config.

    Args:
        input_yaml: Original args.yaml
        output_yaml: Output YAML with updated paths
        data_yaml: Path to dataset YAML
        project_dir: Project output directory
    """
    # Load original config
    with open(input_yaml) as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {input_yaml}")
    print(f"\nOriginal paths:")
    print(f"  data: {config.get('data')}")
    print(f"  project: {config.get('project')}")
    print(f"  save_dir: {config.get('save_dir')}")

    # Update only path-related fields
    config['data'] = str(Path(data_yaml).absolute())
    config['project'] = str(Path(project_dir).absolute())

    # Update save_dir if it exists
    if 'save_dir' in config:
        # Reconstruct save_dir based on project
        config['save_dir'] = str(Path(project_dir).absolute() / 'train')

    print(f"\nUpdated paths:")
    print(f"  data: {config['data']}")
    print(f"  project: {config['project']}")
    if 'save_dir' in config:
        print(f"  save_dir: {config['save_dir']}")

    # Save updated config
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Saved updated config to: {output_yaml}")
    print(f"\n📝 All training parameters preserved")

    return config


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Update YAML paths')
    parser.add_argument('--input', type=str, default='/DATA1/yunzhu/SSL/args.yaml')
    parser.add_argument('--output', type=str, default='/DATA1/yunzhu/SSL/args_updated.yaml')
    parser.add_argument('--data', type=str, default='/DATA1/yunzhu/SSL/valve_detection.yaml')
    parser.add_argument('--project', type=str, default='/DATA1/yunzhu/SSL/valve_training_results')

    args = parser.parse_args()

    update_yaml_paths(
        input_yaml=args.input,
        output_yaml=args.output,
        data_yaml=args.data,
        project_dir=args.project
    )
