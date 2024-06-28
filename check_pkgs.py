import pkg_resources

print("Installed packages:")
for pkg in pkg_resources.working_set:
    print(f"{pkg.project_name} - {pkg.version}")
