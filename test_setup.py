import setup_libs as libs

print('✓ setup_libs imported successfully')
print('\nAvailable packages:')
availability = libs.availability()
for key, value in availability.items():
    print(f'  {key}: {value}')
