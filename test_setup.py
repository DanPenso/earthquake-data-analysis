import setup_libs as libs

print('✓ setup_libs module imported successfully')
print('\nAvailable packages:')
availability = libs.availability()
for key, value in availability.items():
    print(f'  {key}: {value}')
