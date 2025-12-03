import sys
print("Python version:", sys.version)

# Read the setup_libs file and check for 'def availability'
with open('setup_libs.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if 'def availability' in content:
        print("✓ 'def availability' found in setup_libs.py source")
        idx = content.find('def availability')
        print(f"  Location: character position {idx}")
        print(f"  Context: ...{content[max(0,idx-50):idx+150]}...")
    else:
        print("✗ 'def availability' NOT found in setup_libs.py source")

# Now try to import
print("\nAttempting to import setup_libs...")
try:
    import earthquake as setup_libs
    print("✓ earthquake (setup) imported successfully")
    print(f"  Module file: {setup_libs.__file__}")
    print(f"  Module dict keys: {sorted([x for x in dir(setup_libs) if not x.startswith('_')])}")
    
    # Check if availability exists
    if hasattr(setup_libs, 'availability'):
        print("✓ earthquake.availability exists")
        result = setup_libs.availability()
        print(f"  Result: {result}")
    else:
        print("✗ setup_libs.availability DOES NOT EXIST")
        
        # Check for similar names
        similar = [x for x in dir(setup_libs) if 'avail' in x.lower()]
        if similar:
            print(f"  Similar attributes: {similar}")
        else:
            print("  No similar attributes found")
            
except Exception as e:
    print(f"✗ Error importing setup_libs: {e}")
    import traceback
    traceback.print_exc()
